#include "embedding_utilities.h"
#include <cmath>
#include <limits>

namespace {
// Read only data[].index and data[].embedding; do not build a JSON tree.
class embedding_reader : public nlohmann::json_sax<nlohmann::json> {
    enum class scope { root, data, entry, embedding, ignored };
    std::vector<scope> stack;
    std::string field;
    span<float> output;
    size_t count;
    size_t dimensions;
    size_t row_width = 0;
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
        if (row_width == dimensions || !std::isfinite(number) ||
            std::abs(number) > std::numeric_limits<float>::max()) return false;
        output[indices.size() * dimensions + row_width++] = static_cast<float>(number);
        return true;
    }

public:
    embedding_reader(span<float> output, size_t count)
        : output(output), count(count), dimensions(output.size() / count), seen(count, false) {
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
            row_width = 0;
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
            if (!has_index || !has_embedding || row_width != dimensions) return false;
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

void chunk_memory(const docs_container& local_docs, const int thread_id, vector<string_view>& chunks) {
    chunks.clear();
    constexpr size_t max_chunk_bytes = 512;
    const auto& text = local_docs.doc_mem;
    const auto& offsets = local_docs.worker_offsets;
    const size_t worker = static_cast<size_t>(thread_id);
    const size_t worker_start = offsets.at(worker);
    const size_t worker_end = worker + 1 < offsets.size() ? offsets[worker + 1] : text.size();
    if (worker_start == worker_end) return;

    // Find the first document by summing sizes up to this worker's byte offset.
    size_t doc_index = 0;
    size_t start = 0;
    while (start < worker_start) {
        start += local_docs.doc_sizes[doc_index++];
    }

    while (start < worker_end) {
        const size_t doc_end = start + local_docs.doc_sizes[doc_index++];
        while (start < doc_end) {
            size_t end = start + std::min(max_chunk_bytes, doc_end - start);

            // Keep a multibyte UTF-8 character entirely in the next chunk.
            if (end < doc_end) {
                while (end > start && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) {
                    --end;
                }
                if (end == start) {
                    throw std::runtime_error("Invalid UTF-8 at chunk boundary");
                }
            }

            chunks.emplace_back(text.data() + start, end - start);
            start = end;
        }
    }
}

void embed_chunks(const vector<string_view>& chunks, span<float>& embeddings) {
    if (chunks.empty() ? !embeddings.empty()
                       : embeddings.size() < chunks.size() || embeddings.size() % chunks.size() != 0) {
        throw std::invalid_argument("Embedding slice must contain one row per chunk");
    }
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

    embedding_reader reader(embeddings, chunks.size());
    if (!nlohmann::json::sax_parse(response->body, &reader) || !reader.finish()) {
        throw std::runtime_error("Invalid embedding response: unexpected row count, index, or width");
    }
}

void resize_embeddings(vector<float>& embeddings, vector<size_t>& chunk_sizes, size_t embedding_dim) {
    size_t total_chunk_count = 0;
    for (const auto& size : chunk_sizes) {
        total_chunk_count += size;
    }
    embeddings.resize(total_chunk_count * embedding_dim);
}

void create_thread_embeddings_span(vector<float> &embeddings, const vector<size_t>& chunk_sizes, size_t embedding_dim, int thread_id, span<float>& thread_embeddings) {
    // calculate your starting and ending indices for each thread
    // * embedding_dim accounts for embedding dimensions, since embeddings are laid out as contiguous blocks of floats per chunk.
    size_t start_index = 0;
    for (int i = 0; i < thread_id; ++i) {
        start_index += chunk_sizes[i] * embedding_dim;
    }
    size_t end_index = start_index + chunk_sizes[thread_id] * embedding_dim;
    thread_embeddings = span<float>(embeddings.data() + start_index, end_index - start_index);
}
