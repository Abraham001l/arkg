#include "graph_utilities.h"
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <stdexcept>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <type_traits>

void build_adjacency(matrix_rep_adjacency &adj, vector<float> &embeddings, float threshold) {
    if (embeddings.size() % D != 0) {
        throw std::invalid_argument("Embedding data must contain complete vectors");
    }
    if (!std::isfinite(threshold) || threshold < -1.0f || threshold > 1.0f) {
        throw std::invalid_argument("Cosine threshold must be between -1 and 1");
    }
    const size_t cnt = embeddings.size() / D;

    adj.adjacency.assign(cnt, vector<bool>(cnt, false));
    if (cnt == 0) return;

    // build vector of embedding norms
    vector<float> norms(cnt);
    norm(embeddings, norms);

    // calculating numerator for cosine similarity
    vector<float> numerator(cnt * cnt);
    matrix_mul(embeddings, numerator, D);

    // zero-norm vectors remain isolated. Compute each undirected edge once.
    for (size_t i = 0; i < cnt; ++i) {
        if (norms[i] == 0.0f) continue;
        for (size_t j = i + 1; j < cnt; ++j) {
            if (norms[j] == 0.0f) continue;
            const double denominator = static_cast<double>(norms[i]) * norms[j];
            const double similarity = numerator[i * cnt + j] / denominator;
            const bool connected = std::isfinite(similarity) && similarity > threshold;
            adj.adjacency[i][j] = connected;
            adj.adjacency[j][i] = connected;
        }
    }
}

void build_csr(csr_graph &csr, const matrix_rep_adjacency &adj) {
    const size_t count = adj.adjacency.size();
    for (const auto& row : adj.adjacency) {
        if (row.size() != count) {
            throw std::invalid_argument("Adjacency matrix must be square");
        }
    }

    // pass 1, count edges and build the prefix sum directly in offsets.
    csr.offsets.resize(count + 1);
    csr.offsets[0] = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto& row = adj.adjacency[i];
        const size_t degree = static_cast<size_t>(std::count(row.begin(), row.end(), true));
        if (degree > csr.neighbors.max_size() - csr.offsets[i]) {
            throw std::length_error("CSR neighbor array is too large");
        }
        csr.offsets[i + 1] = csr.offsets[i] + degree;
    }

    // allocate once (or reuse existing capacity), with no growth during filling.
    csr.neighbors.resize(csr.offsets[count]);

    // pass 2, fill disjoint row ranges, keeping neighbor IDs sorted.
    for (size_t i = 0; i < count; ++i) {
        if (csr.offsets[i] == csr.offsets[i + 1]) continue;
        const auto& row = adj.adjacency[i];
        size_t output = csr.offsets[i];
        for (size_t j = 0; j < count; ++j) {
            if (row[j]) {
                csr.neighbors[output++] = j;
            }
        }
    }
}

void norm(vector<float> &embeddings, vector<float> &norms) {
    const size_t cnt = embeddings.size() / D;
    norms.resize(cnt);
    for (size_t i = 0; i < cnt; ++i) {
        norms[i] = cblas_snrm2(D, embeddings.data() + i * D, 1);
    }
}

void matrix_mul(vector<float> &embeddings, vector<float> &result, size_t d) {
    const size_t cnt = embeddings.size() / d;

    // perform matrix multiplication using cblas
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        cnt,
        cnt,
        d,
        1.0f,
        embeddings.data(),
        d,
        embeddings.data(),
        d,
        0.0f,
        result.data(),
        cnt
    );
}

string save_graph(const csr_graph& csr, const vector<float>& embeddings,
                  const docs_container& docs, const vector<size_t>& doc_indices,
                  const vector<size_t>& chunk_byte_offsets, const vector<size_t>& chunk_byte_lengths,
                  const string& file_host_url, float threshold) {
    static_assert(sizeof(float) == 4 && numeric_limits<float>::is_iec559);
    int rank, rank_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    httplib::Client client(file_host_url);

    // Every rank reaches the same checkpoint, including when one upload fails.
    string error;
    const auto check_all_ranks = [&] {
        const int failed = !error.empty();
        int any_failed;
        MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (any_failed) {
            throw runtime_error(error.empty() ? "Another rank failed to save the graph" : error);
        }
    };
    const auto require_success = [](const httplib::Result& response) {
        if (!response) {
            throw runtime_error("Graph upload failed: " + httplib::to_string(response.error()));
        }
        if (response->status != 200) {
            throw runtime_error("Graph upload returned HTTP " + to_string(response->status) +
                                ": " + response->body);
        }
    };

    const size_t nodes = doc_indices.size();
    try {
        if (embeddings.size() % D != 0 || embeddings.size() / D != nodes ||
            chunk_byte_offsets.size() != nodes || chunk_byte_lengths.size() != nodes ||
            docs.doc_paths.size() != docs.doc_sizes.size() || csr.offsets.size() != nodes + 1 ||
            csr.offsets.front() != 0 || csr.offsets.back() != csr.neighbors.size() ||
            !is_sorted(csr.offsets.begin(), csr.offsets.end()) ||
            !isfinite(threshold) || threshold < -1 || threshold > 1) {
            throw runtime_error("Graph arrays are inconsistent");
        }
        for (size_t i = 0; i < nodes; ++i) {
            if (doc_indices[i] >= docs.doc_paths.size()) throw runtime_error("Invalid document index");
            const size_t size = docs.doc_sizes[doc_indices[i]];
            if (chunk_byte_offsets[i] > size || chunk_byte_lengths[i] > size - chunk_byte_offsets[i]) {
                throw runtime_error("Chunk is outside its document text");
            }
        }
        for (size_t neighbor : csr.neighbors) {
            if (neighbor >= nodes) throw runtime_error("Invalid neighbor index");
        }
    } catch (const exception& failure) {
        error = failure.what();
    }
    check_all_ranks();

    // Keep the graph layout here; the file host only creates directories and writes bytes.
    string build_id;
    if (rank == 0) {
        try {
            const nlohmann::json options{{"path", ".arkg-graphs/build"}, {"unique", true}};
            const auto response = client.Post("/create_directory", options.dump(), "application/json");
            require_success(response);
            const auto path = nlohmann::json::parse(response->body).at("path").get<string>();
            build_id = filesystem::path(path).filename().string();
        } catch (const exception& failure) {
            error = failure.what();
        }
    }
    check_all_ranks();
    int id_length = static_cast<int>(build_id.size());
    MPI_Bcast(&id_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    build_id.resize(id_length);
    MPI_Bcast(build_id.data(), id_length, MPI_CHAR, 0, MPI_COMM_WORLD);

    const string shard_id = "shard-" + to_string(rank);
    const string build_path = ".arkg-graphs/" + build_id;
    const string shard_path = build_path + "/shards/" + shard_id;
    const auto upload = [&](const string& file, const string& bytes) {
        const auto endpoint = httplib::append_query_params("/save_file", {{"path", file}});
        const auto response = client.Post(endpoint, bytes, "application/octet-stream");
        require_success(response);
        if (nlohmann::json::parse(response->body).at("byte_length").get<size_t>() != bytes.size()) {
            throw runtime_error("Saved file length does not match uploaded bytes");
        }
    };
    // Encode fixed-width little-endian values, independent of size_t and struct layout.
    const auto upload_array = [&]<typename T>(const string& file, const vector<T>& values) {
        constexpr size_t width = is_floating_point_v<T> ? 4 : 8;
        string bytes;
        if (values.size() > bytes.max_size() / width) throw length_error("Graph array is too large");
        bytes.resize(values.size() * width);
        for (size_t i = 0; i < values.size(); ++i) {
            uint64_t bits;
            if constexpr (is_floating_point_v<T>) bits = bit_cast<uint32_t>(values[i]);
            else bits = static_cast<uint64_t>(values[i]);
            for (size_t byte = 0; byte < width; ++byte) {
                bytes[i * width + byte] = static_cast<char>((bits >> (byte * 8)) & 0xff);
            }
        }
        upload(shard_path + '/' + file, bytes);
    };
    try {
        const nlohmann::json directory{{"path", shard_path}};
        require_success(client.Post("/create_directory", directory.dump(), "application/json"));
        upload_array("offsets.bin", csr.offsets);
        upload_array("neighbors.bin", csr.neighbors);
        upload_array("embeddings.bin", embeddings);
        upload_array("doc_indices.bin", doc_indices);
        upload_array("chunk_byte_offsets.bin", chunk_byte_offsets);
        upload_array("chunk_byte_lengths.bin", chunk_byte_lengths);
        upload(shard_path + "/doc_paths.json", nlohmann::json(docs.doc_paths).dump());
    } catch (const exception& failure) {
        error = failure.what();
    }
    check_all_ranks();

    const array<uint64_t, 3> counts{nodes, csr.neighbors.size(), docs.doc_paths.size()};
    vector<uint64_t> shard_counts(static_cast<size_t>(rank_count) * counts.size());
    MPI_Gather(counts.data(), 3, MPI_UINT64_T, shard_counts.data(), 3, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        try {
            nlohmann::json manifest{
                {"format_version", 1}, {"build_id", build_id},
                {"embedding_model", "bge-small-en-v1.5-q8_0.gguf"}, {"embedding_dimension", D},
                {"byte_order", "little"}, {"index_type", "uint64"}, {"embedding_type", "float32"},
                {"similarity", "cosine"}, {"threshold", threshold},
                {"neighbor_ids", "shard_local"}, {"document_indices", "shard_local"},
                {"chunk_offsets", "document_extracted_text_bytes"},
                {"source_url", file_host_url}, {"shards", nlohmann::json::array()}
            };
            for (int shard = 0; shard < rank_count; ++shard) {
                const size_t base = static_cast<size_t>(shard) * 3;
                manifest["shards"].push_back({
                    {"id", "shard-" + to_string(shard)},
                    {"path", "shards/shard-" + to_string(shard)},
                    {"node_count", shard_counts[base]}, {"neighbor_count", shard_counts[base + 1]},
                    {"document_count", shard_counts[base + 2]}
                });
            }
            manifest["files"] = {
                {"offsets", "offsets.bin"}, {"neighbors", "neighbors.bin"},
                {"embeddings", "embeddings.bin"}, {"doc_indices", "doc_indices.bin"},
                {"chunk_byte_offsets", "chunk_byte_offsets.bin"},
                {"chunk_byte_lengths", "chunk_byte_lengths.bin"}, {"doc_paths", "doc_paths.json"}
            };
            // All ranks confirmed their uploads before the manifest makes this build visible.
            upload(build_path + "/manifest.json", manifest.dump(2));
        } catch (const exception& failure) {
            error = failure.what();
        }
    }
    check_all_ranks();
    return build_id;
}
