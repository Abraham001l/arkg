#pragma once
#include "doc_utilities.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string_view>
#include <vector>
using namespace std;

void chunk_memory(work_container& w, std::vector<std::string_view> &chunks);

// replaces embeddings with contiguous rows in chunk order; allocates as needed.
// for nonempty input, row width is embeddings.size() / chunks.size().
// throws on request/response errors, leaving embeddings empty.
void embed_chunks(const std::vector<std::string_view> &chunks, vector<float> &embeddings);
