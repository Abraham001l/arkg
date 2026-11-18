#pragma once
#include "doc_utilities.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string_view>
#include <vector>
using namespace std;

// local_doc_indices[i] identifies chunks[i]'s document in local_docs.doc_paths.
// Byte offsets are relative to that document's text, and lengths are in bytes.
void chunk_memory(const docs_container& local_docs, const int thread_id,
                  vector<string_view>& chunks, vector<size_t>& local_doc_indices,
                  vector<size_t>& local_chunk_byte_offsets, vector<size_t>& local_chunk_byte_lengths);

// replaces embeddings with contiguous rows in chunk order; allocates as needed.
// for nonempty input, row width is embeddings.size() / chunks.size().
// throws on request/response errors, leaving embeddings empty.
void embed_chunks(const vector<string_view> &chunks, span<float> &embeddings);

void resize_embeddings(vector<float> &embeddings, vector<size_t> &chunk_sizes, size_t embedding_dim);

void create_thread_embeddings_span(vector<float> &embeddings, const vector<size_t> &chunk_sizes, size_t embedding_dim, int thread_id, span<float> &thread_embeddings);

void resize_doc_indices(vector<size_t> &doc_indices, const vector<size_t> &chunk_sizes);

void copy_local_doc_indices_to_global(vector<size_t> &doc_indices, const vector<size_t> &local_doc_indices, const vector<size_t> &chunk_sizes, int thread_id);

void resize_chunk_byte_offsets(vector<size_t>& chunk_byte_offsets, const vector<size_t>& chunk_sizes);

void resize_chunk_byte_lengths(vector<size_t>& chunk_byte_lengths, const vector<size_t>& chunk_sizes);

void copy_local_chunk_byte_offsets_to_global(vector<size_t>& chunk_byte_offsets,
    const vector<size_t>& local_chunk_byte_offsets, const vector<size_t>& chunk_sizes, int thread_id);

void copy_local_chunk_byte_lengths_to_global(vector<size_t>& chunk_byte_lengths,
    const vector<size_t>& local_chunk_byte_lengths, const vector<size_t>& chunk_sizes, int thread_id);
