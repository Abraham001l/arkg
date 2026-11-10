#pragma once
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

struct docs_container {
    size_t doc_mem_offset;
    size_t doc_sizes_offset;
    size_t rank_offsets_offset;
    span<char> doc_mem;
    span<size_t> doc_sizes;
    span<size_t> rank_offsets;
};

struct work_container {
    size_t work_mem_size;
    span<char> work_mem;
    unique_ptr<char[]> owned_memory; // Keeps work_mem's bytes alive; frees them automatically.
};

bool pick_doc_retriever(const string& doc_path, int my_rank, int& master_rank);

void populate_docs_container(docs_container& d, char* data_ptr, const vector<string>& doc_paths);

void calc_rank_offsets(docs_container& d, char* data_ptr, size_t world_size);

void distribute_docs_to_ranks(docs_container& d, char* data_ptr, size_t world_size, int my_rank);

void receive_docs_from_master(work_container& w, int master_rank);

void fill_work_container_for_master_rank(work_container& w, docs_container& d, char* data_ptr, int my_rank);

void free_docs_container(docs_container& d);
