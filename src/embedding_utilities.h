#pragma once
#include "doc_utilities.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string_view>
#include <vector>
using namespace std;

void chunk_memory(const docs_container& local_docs, const int thread_id, vector<string_view>& chunks);

// replaces embeddings with contiguous rows in chunk order; allocates as needed.
// for nonempty input, row width is embeddings.size() / chunks.size().
// throws on request/response errors, leaving embeddings empty.
void embed_chunks(const vector<string_view> &chunks, span<float> &embeddings);

void resize_embeddings(vector<float> &embeddings, vector<size_t> &chunk_sizes, size_t embedding_dim);

void create_thread_embeddings_span(vector<float> &embeddings, const vector<size_t> &chunk_sizes, size_t embedding_dim, int thread_id, span<float> &thread_embeddings);
