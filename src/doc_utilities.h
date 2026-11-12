#pragma once
#include <cstdint>
#include <vector>
#include <fstream>
#include <span>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <system_error>
#include <limits>
#include <memory>
#include "mpi.h"
using namespace std;

constexpr int size_tag = 10;
constexpr int data_tag = 20;
constexpr int doc_sizes_tag = 30;
constexpr size_t one_GB = 1024 * 1024 * 1024;

struct docs_container {
    vector<char> doc_mem;
    vector<size_t> doc_sizes;
    vector<size_t> worker_offsets;
};

bool pick_doc_retriever(const string& doc_path, int my_rank, int& master_rank);

void populate_docs_container(docs_container& global_docs, const vector<string>& doc_paths);

void calc_offsets(docs_container& d, size_t world_size);

void distribute_docs_to_ranks(const docs_container& global_docs, docs_container& local_docs,
                              int world_size, int my_rank);

void receive_docs_from_master(docs_container& local_docs, int master_rank);
