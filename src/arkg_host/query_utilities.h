#pragma once
#include <httplib.h>
#include <span>
#include <vector>
#include <string>
using namespace std;

struct graph {
    vector<size_t> neighbors;
    vector<size_t> offsets;
    vector<float> embeddings;
    vector<size_t> doc_indices;
    vector<size_t> chunk_byte_offsets;
    vector<size_t> chunk_byte_lengths;
    vector<string> doc_paths;
};

void populate_subgraph_paths(httplib::Client& client, const string& build_id,
                             vector<string>& subgraph_paths);

void calc_rank_subgraph_indices(size_t world_size, size_t num_subgraphs, vector<size_t>& rank_subgraph_indices);

void distribute_subgraph_paths_to_ranks(const vector<string>& global_subgraph_paths,
                                        const vector<size_t>& rank_subgraph_indices,
                                        vector<string>& local_subgraph_paths, int my_rank);

void receive_subgraph_paths_from_master(vector<string>& local_subgraph_paths, int master_rank);

void calc_thread_subgraph_indices(size_t thread_count, size_t num_subgraphs, vector<size_t>& thread_subgraph_indices);

void extract_view_of_local_subgraph_paths(const vector<string>& local_subgraph_paths, const vector<size_t>& thread_subgraph_indices,
                                          int thread_id, span<const string>& thread_subgraph_paths);

void readin_local_subgraphs(span<const string> thread_subgraph_paths,
                            vector<graph>& local_subgraphs,
                            httplib::Client& graph_file_client);

void run_query_host(int rank, const string& graph_host_url, const string& source_host_url,
                    int query_port, const vector<string>& local_subgraph_paths);
