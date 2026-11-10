#include "embedding_utilities.h"
#include <cmath>
#include <limits>

namespace {
// Read only data[].index and data[].embedding; do not build a JSON tree.
class embedding_reader : public nlohmann::json_sax<nlohmann::json> {
    enum class scope { root, data, entry, embedding, ignored };
    std::vector<scope> stack;
    std::string field;
    std::vector<float>& output;
    size_t count;
    size_t dimensions = 0;
    size_t row_start = 0;
    size_t index = 0;
    bool has_index = false;
    bool has_embedding = false;
    bool has_data = false;
    std::vector<size_t> indices;
    std::vector<bool> seen;

    scope current() const { return stack.empty() ? scope::ignored : stack.back(); }

    bool scalar_allowed() const {
        return current() != scope::embedding && current() != scope::data &&
            !(current() == scope::entry && (field == "index" || field == "embedding")) &&
            !(current() == scope::root && field == "data");
    }

    bool value(double number) {
        if (current() != scope::embedding) return scalar_allowed();
        if (!std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max()) return false;
        output.push_back(static_cast<float>(number));
        return true;
    }

public:
    embedding_reader(std::vector<float>& output, size_t count)
        : output(output), count(count), seen(count, false) {
        indices.reserve(count);
    }

    bool null() override { return scalar_allowed(); }
    bool boolean(bool) override { return scalar_allowed(); }
    bool string(string_t&) override { return scalar_allowed(); }
    bool binary(binary_t&) override { return false; }
    bool number_float(number_float_t n, const string_t&) override { return value(n); }
    bool number_integer(number_integer_t n) override {
        if (current() == scope::entry && field == "index") {
            return n >= 0 && number_unsigned(static_cast<number_unsigned_t>(n));
        }
        return value(static_cast<double>(n));
    }
    bool number_unsigned(number_unsigned_t n) override {
        if (current() == scope::entry && field == "index") {
            if (has_index || n >= count || seen[n]) return false;
            index = static_cast<size_t>(n);
            has_index = true;
            return true;
        }
        return value(static_cast<double>(n));
    }
    bool key(string_t& name) override { field = name; return true; }
    bool start_object(size_t) override {
        if (stack.empty()) {
            stack.push_back(scope::root);
        } else if (current() == scope::data) {
            if (indices.size() == count) return false;
            row_start = output.size();
            has_index = has_embedding = false;
            stack.push_back(scope::entry);
        } else {
            if (!scalar_allowed()) return false;
            stack.push_back(scope::ignored);
        }
        return true;
    }
    bool end_object() override {
        if (current() == scope::entry) {
            if (!has_index || !has_embedding) return false;
            const size_t width = output.size() - row_start;
            if (width == 0 || (dimensions != 0 && width != dimensions)) return false;
            if (dimensions == 0) {
                dimensions = width;
                if (count > output.max_size() / dimensions) return false;
                // Once the first row tells us its width, allocate for the whole batch.
                output.reserve(count * dimensions);
            }
            seen[index] = true;
            indices.push_back(index);
        }
        stack.pop_back();
        return true;
    }
    bool start_array(size_t) override {
        if (stack.empty()) return false;
        if (current() == scope::root && field == "data") {
            if (has_data) return false;
            has_data = true;
            stack.push_back(scope::data);
        } else if (current() == scope::entry && field == "embedding") {
            if (has_embedding) return false;
            has_embedding = true;
            stack.push_back(scope::embedding);
        } else {
            if (!scalar_allowed()) return false;
            stack.push_back(scope::ignored);
        }
        return true;
    }
    bool end_array() override { stack.pop_back(); return true; }
    bool parse_error(size_t, const std::string&, const nlohmann::detail::exception&) override {
        return false;
    }

    bool finish() {
        if (!has_data || indices.size() != count) return false;
        // Usually already ordered; otherwise rearrange rows in place using index.
        for (size_t i = 0; i < count; ++i) {
            while (indices[i] != i) {
                const size_t destination = indices[i];
                std::swap_ranges(output.begin() + i * dimensions,
                                 output.begin() + (i + 1) * dimensions,
                                 output.begin() + destination * dimensions);
                std::swap(indices[i], indices[destination]);
            }
        }
        return true;
    }
};
}

void chunk_memory(work_container& w, std::vector<std::string_view> &chunks) {
    chunks.clear();
    constexpr size_t max_chunk_bytes = 512;
    size_t start = 0;

    while (start < w.work_mem.size()) {
        size_t end = start + std::min(max_chunk_bytes, w.work_mem.size() - start);

        // Keep a multibyte UTF-8 character entirely in the next chunk.
        if (end < w.work_mem.size()) {
            while (end > start &&
                   (static_cast<unsigned char>(w.work_mem[end]) & 0xC0) == 0x80) {
                --end;
            }
            if (end == start) {
                throw std::runtime_error("Invalid UTF-8 at chunk boundary");
            }
        }

        chunks.emplace_back(w.work_mem.data() + start, end - start);
        start = end;
    }
}

void embed_chunks(const std::vector<std::string_view> &chunks, vector<float> &embeddings) {
    embeddings.clear();
    if (chunks.empty()) return;

    httplib::Client client("http://127.0.0.1:8080");
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(120, 0);

    std::string request;
    {
        nlohmann::json body = {
            {"model", "bge-small-en-v1.5-q8_0.gguf"},
            {"input", chunks},
            {"encoding_format", "float"}
        };
        request = body.dump();
    }
    auto response = client.Post("/v1/embeddings", request, "application/json");
    if (!response) {
        throw std::runtime_error("Embedding request failed: " + httplib::to_string(response.error()));
    }
    if (response->status != 200) {
        throw std::runtime_error("Embedding server returned HTTP " +
                                 std::to_string(response->status) + ": " + response->body);
    }

    try {
        embedding_reader reader(embeddings, chunks.size());
        if (!nlohmann::json::sax_parse(response->body, &reader) || !reader.finish()) {
            throw std::runtime_error("Invalid embedding response: expected one indexed, equal-sized vector per chunk");
        }
    } catch (...) {
        embeddings.clear();
        throw;
    }
}
