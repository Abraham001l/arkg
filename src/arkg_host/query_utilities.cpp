#include "query_utilities.h"
#include <mpi.h>
#include <omp.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

namespace {
constexpr int subgraph_paths_size_tag = 10;
constexpr int subgraph_paths_data_tag = 20;
constexpr int query_results_size_tag = 30;
constexpr int query_results_data_tag = 40;
constexpr size_t embedding_dimension = 384;

struct query_request {
    vector<float> embedding;
    size_t top_k;
};

struct query_inbox {
    mutex sync_mutex;
    condition_variable query_ready;
    condition_variable response_ready;
    condition_variable request_slot_ready;
    string request;
    string response;
    bool request_waiting = false;
    bool request_in_progress = false;
    bool response_available = false;
};

struct local_query_gate {
    mutex sync_mutex;
    condition_variable query_ready;
    condition_variable workers_finished;
    shared_ptr<const query_request> request;
    size_t generation = 0;
    size_t completed_workers = 0;
    vector<nlohmann::json> worker_results;
};

query_request parse_query_request(const string& body) {
    const auto json = nlohmann::json::parse(body);
    query_request request;
    request.embedding = json.at("embedding").get<vector<float>>();
    request.top_k = json.value("top_k", size_t{1});

    if (request.embedding.size() != embedding_dimension) {
        throw invalid_argument("embedding must contain exactly " +
                               to_string(embedding_dimension) + " floats");
    }
    if (request.top_k == 0) {
        throw invalid_argument("top_k must be greater than zero");
    }

    double norm_squared = 0.0;
    for (const float value : request.embedding) {
        if (!isfinite(value)) {
            throw invalid_argument("embedding values must be finite");
        }
        norm_squared += static_cast<double>(value) * value;
    }
    if (norm_squared == 0.0) {
        throw invalid_argument("embedding must have a nonzero norm");
    }
    return request;
}

void register_query_endpoint(httplib::Server& server, query_inbox& inbox) {
    server.Post("/query", [&inbox](const httplib::Request& request,
                                    httplib::Response& response) {
        try {
            // Reject malformed requests before any rank enters a collective operation.
            parse_query_request(request.body);

            unique_lock lock(inbox.sync_mutex);
            inbox.request_slot_ready.wait(lock, [&] {
                return !inbox.request_in_progress;
            });

            inbox.request = request.body;
            inbox.request_waiting = true;
            inbox.request_in_progress = true;
            inbox.query_ready.notify_one();

            inbox.response_ready.wait(lock, [&] {
                return inbox.response_available;
            });
            response.set_content(inbox.response, "application/json");
            inbox.response_available = false;
            inbox.request_in_progress = false;
            lock.unlock();
            inbox.request_slot_ready.notify_one();
        } catch (const exception& error) {
            response.status = 400;
            response.set_content(nlohmann::json{{"error", error.what()}}.dump(),
                                 "application/json");
        }
    });
}

string wait_for_query(query_inbox& inbox) {
    unique_lock lock(inbox.sync_mutex);
    inbox.query_ready.wait(lock, [&] { return inbox.request_waiting; });
    inbox.request_waiting = false;
    return inbox.request;
}

void finish_query(query_inbox& inbox, string response) {
    {
        lock_guard lock(inbox.sync_mutex);
        inbox.response = move(response);
        inbox.response_available = true;
    }
    inbox.response_ready.notify_one();
}

string broadcast_query(int rank, query_inbox* inbox) {
    string body;
    if (rank == 0) {
        body = wait_for_query(*inbox);
    }

    uint64_t byte_count = body.size();
    MPI_Bcast(&byte_count, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (byte_count > static_cast<uint64_t>(numeric_limits<int>::max())) {
        throw length_error("Query request is too large for MPI");
    }

    if (rank != 0) {
        body.resize(static_cast<size_t>(byte_count));
    }
    MPI_Bcast(body.data(), static_cast<int>(byte_count), MPI_CHAR, 0, MPI_COMM_WORLD);
    return body;
}

void publish_query(local_query_gate& gate, shared_ptr<const query_request> request) {
    {
        lock_guard lock(gate.sync_mutex);
        gate.request = move(request);
        gate.completed_workers = 0;
        for (auto& results : gate.worker_results) {
            results = nlohmann::json::array();
        }
        ++gate.generation;
    }
    gate.query_ready.notify_all();
}

vector<nlohmann::json> wait_for_local_results(local_query_gate& gate) {
    unique_lock lock(gate.sync_mutex);
    gate.workers_finished.wait(lock, [&] {
        return gate.completed_workers == gate.worker_results.size();
    });
    return gate.worker_results;
}

struct scored_node {
    float similarity;
    size_t node;
};

struct lower_similarity {
    bool operator()(const scored_node& left, const scored_node& right) const {
        return left.similarity > right.similarity;
    }
};

nlohmann::json query_subgraph(const graph& subgraph, const string& subgraph_path,
                              const query_request& request, int rank) {
    const size_t node_count = subgraph.offsets.size() - 1;
    priority_queue<scored_node, vector<scored_node>, lower_similarity> best;

    double query_norm_squared = 0.0;
    for (const float value : request.embedding) {
        query_norm_squared += static_cast<double>(value) * value;
    }
    const double query_norm = sqrt(query_norm_squared);

    for (size_t node = 0; node < node_count; ++node) {
        const float* embedding = subgraph.embeddings.data() + node * embedding_dimension;
        double dot_product = 0.0;
        double node_norm_squared = 0.0;
        for (size_t dimension = 0; dimension < embedding_dimension; ++dimension) {
            dot_product += static_cast<double>(request.embedding[dimension]) * embedding[dimension];
            node_norm_squared += static_cast<double>(embedding[dimension]) * embedding[dimension];
        }
        if (node_norm_squared == 0.0) continue;

        const float similarity = static_cast<float>(
            dot_product / (query_norm * sqrt(node_norm_squared)));
        if (!isfinite(similarity)) continue;

        if (best.size() < request.top_k) {
            best.push({similarity, node});
        } else if (similarity > best.top().similarity) {
            best.pop();
            best.push({similarity, node});
        }
    }

    vector<scored_node> ordered;
    ordered.reserve(best.size());
    while (!best.empty()) {
        ordered.push_back(best.top());
        best.pop();
    }
    sort(ordered.begin(), ordered.end(), [](const scored_node& left,
                                             const scored_node& right) {
        return left.similarity > right.similarity;
    });

    nlohmann::json matches = nlohmann::json::array();
    for (const auto [similarity, node] : ordered) {
        const size_t doc_index = subgraph.doc_indices[node];
        matches.push_back({
            {"node", node},
            {"similarity", similarity},
            {"doc_path", subgraph.doc_paths[doc_index]},
            {"byte_offset", subgraph.chunk_byte_offsets[node]},
            {"byte_length", subgraph.chunk_byte_lengths[node]},
        });
    }

    return {
        {"rank", rank},
        {"subgraph_path", subgraph_path},
        {"matches", move(matches)},
    };
}

void run_graph_worker(size_t worker_id, int rank, span<const string> subgraph_paths,
                      vector<graph>& subgraphs, local_query_gate& gate) {
    size_t last_generation = 0;
    while (true) {
        shared_ptr<const query_request> request;
        size_t generation;
        {
            unique_lock lock(gate.sync_mutex);
            gate.query_ready.wait(lock, [&] {
                return gate.generation != last_generation;
            });
            request = gate.request;
            generation = gate.generation;
        }

        nlohmann::json results = nlohmann::json::array();
        try {
            for (size_t i = 0; i < subgraphs.size(); ++i) {
                results.push_back(query_subgraph(subgraphs[i], subgraph_paths[i],
                                                 *request, rank));
            }
        } catch (const exception& error) {
            results.push_back({{"rank", rank}, {"error", error.what()}});
        }

        {
            lock_guard lock(gate.sync_mutex);
            gate.worker_results[worker_id] = move(results);
            ++gate.completed_workers;
            last_generation = generation;
        }
        gate.workers_finished.notify_one();
    }
}

nlohmann::json flatten_worker_results(vector<nlohmann::json> worker_results) {
    nlohmann::json results = nlohmann::json::array();
    for (auto& worker : worker_results) {
        for (auto& result : worker) {
            results.push_back(move(result));
        }
    }
    return results;
}

void send_results_to_master(const nlohmann::json& results) {
    const string bytes = results.dump();
    if (bytes.size() > static_cast<size_t>(numeric_limits<int>::max())) {
        throw length_error("Query result is too large for MPI");
    }
    const int byte_count = static_cast<int>(bytes.size());
    MPI_Send(&byte_count, 1, MPI_INT, 0, query_results_size_tag, MPI_COMM_WORLD);
    MPI_Send(bytes.data(), byte_count, MPI_CHAR, 0, query_results_data_tag, MPI_COMM_WORLD);
}

nlohmann::json collect_rank_results(int world_size, nlohmann::json local_results) {
    for (int rank = 1; rank < world_size; ++rank) {
        int byte_count;
        MPI_Recv(&byte_count, 1, MPI_INT, rank, query_results_size_tag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (byte_count < 0) {
            throw runtime_error("Received an invalid query result size");
        }

        string bytes(static_cast<size_t>(byte_count), '\0');
        MPI_Recv(bytes.data(), byte_count, MPI_CHAR, rank, query_results_data_tag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        auto remote_results = nlohmann::json::parse(bytes);
        for (auto& result : remote_results) {
            local_results.push_back(move(result));
        }
    }
    return local_results;
}

void run_query_coordinator(int rank, const string& source_host_url,
                           local_query_gate& gate, query_inbox* inbox) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    while (true) {
        const string body = broadcast_query(rank, inbox);
        auto request = make_shared<const query_request>(parse_query_request(body));
        publish_query(gate, move(request));

        auto local_results = flatten_worker_results(wait_for_local_results(gate));
        if (rank == 0) {
            auto all_results = collect_rank_results(world_size, move(local_results));
            finish_query(*inbox, nlohmann::json{
                {"source_host_url", source_host_url},
                {"results", move(all_results)},
            }.dump());
        } else {
            send_results_to_master(local_results);
        }
    }
}

string get_graph_file(httplib::Client& client, const string& subgraph_path,
                      const string& filename) {
    const string path = (filesystem::path(subgraph_path) / filename).generic_string();
    auto response = client.Get(
        httplib::append_query_params("/get_file", {{"path", path}}));
    if (!response) {
        throw runtime_error("Failed to read graph file " + path + ": " +
                            httplib::to_string(response.error()));
    }
    if (response->status != 200) {
        throw runtime_error("Graph file " + path + " returned HTTP " +
                            to_string(response->status) + ": " + response->body);
    }
    return move(response->body);
}

vector<size_t> decode_indices(const string& bytes, string_view filename) {
    constexpr size_t width = sizeof(uint64_t);
    if (bytes.size() % width != 0) {
        throw runtime_error(string(filename) + " has an invalid byte length");
    }

    vector<size_t> values(bytes.size() / width);
    for (size_t i = 0; i < values.size(); ++i) {
        uint64_t value = 0;
        for (size_t byte = 0; byte < width; ++byte) {
            value |= static_cast<uint64_t>(
                         static_cast<unsigned char>(bytes[i * width + byte]))
                     << (byte * 8);
        }
        if (value > numeric_limits<size_t>::max()) {
            throw overflow_error(string(filename) + " contains an index that does not fit size_t");
        }
        values[i] = static_cast<size_t>(value);
    }
    return values;
}

vector<float> decode_embeddings(const string& bytes) {
    constexpr size_t width = sizeof(uint32_t);
    if (bytes.size() % width != 0) {
        throw runtime_error("embeddings.bin has an invalid byte length");
    }

    vector<float> embeddings(bytes.size() / width);
    for (size_t i = 0; i < embeddings.size(); ++i) {
        uint32_t bits = 0;
        for (size_t byte = 0; byte < width; ++byte) {
            bits |= static_cast<uint32_t>(
                        static_cast<unsigned char>(bytes[i * width + byte]))
                    << (byte * 8);
        }
        embeddings[i] = bit_cast<float>(bits);
    }
    return embeddings;
}

void validate_graph(const graph& subgraph) {
    if (subgraph.offsets.empty() || subgraph.offsets.front() != 0 ||
        !is_sorted(subgraph.offsets.begin(), subgraph.offsets.end()) ||
        subgraph.offsets.back() != subgraph.neighbors.size()) {
        throw runtime_error("Subgraph CSR arrays are inconsistent");
    }

    const size_t node_count = subgraph.offsets.size() - 1;
    if (node_count > numeric_limits<size_t>::max() / embedding_dimension ||
        subgraph.embeddings.size() != node_count * embedding_dimension ||
        subgraph.doc_indices.size() != node_count ||
        subgraph.chunk_byte_offsets.size() != node_count ||
        subgraph.chunk_byte_lengths.size() != node_count) {
        throw runtime_error("Subgraph node arrays are inconsistent");
    }

    for (const size_t neighbor : subgraph.neighbors) {
        if (neighbor >= node_count) {
            throw runtime_error("Subgraph contains an invalid neighbor index");
        }
    }
    for (const size_t doc_index : subgraph.doc_indices) {
        if (doc_index >= subgraph.doc_paths.size()) {
            throw runtime_error("Subgraph contains an invalid document index");
        }
    }
}
}

void populate_subgraph_paths(httplib::Client& client, const string& build_id,
                             vector<string>& subgraph_paths) {
    if (build_id.empty() || build_id == "." || build_id == ".." ||
        build_id.find_first_of("/\\") != string::npos) {
        throw invalid_argument("Build ID must be a single directory name");
    }

    const auto manifest = nlohmann::json::parse(get_graph_file(client, build_id, "manifest.json"));
    const auto& shards = manifest.at("shards");
    if (!shards.is_array()) throw runtime_error("Graph manifest shards must be an array");

    subgraph_paths.clear();
    subgraph_paths.reserve(shards.size());
    for (const auto& shard : shards) {
        subgraph_paths.push_back(
            (filesystem::path(build_id) / shard.at("path").get<string>()).generic_string());
    }
}

void calc_rank_subgraph_indices(size_t world_size, size_t num_subgraphs,
                                vector<size_t>& rank_subgraph_indices) {
    if (world_size == 0) {
        throw invalid_argument("The number of ranks must be greater than zero");
    }

    rank_subgraph_indices.resize(world_size + 1);

    const size_t subgraphs_per_rank = num_subgraphs / world_size;
    const size_t ranks_with_extra_subgraph = num_subgraphs % world_size;

    size_t next_subgraph = 0;
    for (size_t rank = 0; rank < world_size; ++rank) {
        rank_subgraph_indices[rank] = next_subgraph;
        next_subgraph += subgraphs_per_rank;
        if (rank < ranks_with_extra_subgraph) {
            ++next_subgraph;
        }
    }
    rank_subgraph_indices[world_size] = next_subgraph;
}

void distribute_subgraph_paths_to_ranks(const vector<string>& global_subgraph_paths,
                                        const vector<size_t>& rank_subgraph_indices,
                                        vector<string>& local_subgraph_paths, int my_rank) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (my_rank < 0 || my_rank >= world_size) {
        throw invalid_argument("The distributing rank is outside MPI_COMM_WORLD");
    }

    if (rank_subgraph_indices.size() != static_cast<size_t>(world_size) + 1 ||
        rank_subgraph_indices.front() != 0 ||
        rank_subgraph_indices.back() != global_subgraph_paths.size() ||
        !is_sorted(rank_subgraph_indices.begin(), rank_subgraph_indices.end())) {
        throw invalid_argument("Subgraph rank indices are invalid");
    }

    for (int rank = 0; rank < world_size; ++rank) {
        const size_t start = rank_subgraph_indices[rank];
        const size_t end = rank_subgraph_indices[rank + 1];
        const vector<string> paths(global_subgraph_paths.begin() + start,
                                   global_subgraph_paths.begin() + end);

        if (rank == my_rank) {
            local_subgraph_paths = paths;
            continue;
        }

        const string wire_paths = nlohmann::json(paths).dump();
        if (wire_paths.size() > static_cast<size_t>(numeric_limits<int>::max())) {
            throw length_error("Subgraph path message is too large for MPI");
        }
        const int byte_count = static_cast<int>(wire_paths.size());
        MPI_Send(&byte_count, 1, MPI_INT, rank, subgraph_paths_size_tag, MPI_COMM_WORLD);
        MPI_Send(wire_paths.data(), byte_count, MPI_CHAR, rank,
                 subgraph_paths_data_tag, MPI_COMM_WORLD);
    }
}

void receive_subgraph_paths_from_master(vector<string>& local_subgraph_paths, int master_rank) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (master_rank < 0 || master_rank >= world_size) {
        throw invalid_argument("The master rank is outside MPI_COMM_WORLD");
    }
    if (my_rank == master_rank) {
        throw invalid_argument("The master rank cannot receive from itself");
    }

    int byte_count;
    MPI_Recv(&byte_count, 1, MPI_INT, master_rank, subgraph_paths_size_tag,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (byte_count < 0) {
        throw runtime_error("Received an invalid subgraph path message size");
    }

    string wire_paths(static_cast<size_t>(byte_count), '\0');
    MPI_Recv(wire_paths.data(), byte_count, MPI_CHAR, master_rank,
             subgraph_paths_data_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    local_subgraph_paths = nlohmann::json::parse(wire_paths).get<vector<string>>();
}

void calc_thread_subgraph_indices(size_t thread_count, size_t num_subgraphs,
                                  vector<size_t>& thread_subgraph_indices) {
    if (thread_count == 0) {
        throw invalid_argument("The number of threads must be greater than zero");
    }

    thread_subgraph_indices.resize(thread_count + 1);

    const size_t subgraphs_per_thread = num_subgraphs / thread_count;
    const size_t threads_with_extra_subgraph = num_subgraphs % thread_count;

    size_t next_subgraph = 0;
    for (size_t thread = 0; thread < thread_count; ++thread) {
        thread_subgraph_indices[thread] = next_subgraph;
        next_subgraph += subgraphs_per_thread;
        if (thread < threads_with_extra_subgraph) {
            ++next_subgraph;
        }
    }
    thread_subgraph_indices[thread_count] = next_subgraph;
}

void extract_view_of_local_subgraph_paths(const vector<string>& local_subgraph_paths,
                                          const vector<size_t>& thread_subgraph_indices,
                                          int thread_id,
                                          span<const string>& thread_subgraph_paths) {
    if (thread_id < 0 ||
        static_cast<size_t>(thread_id) + 1 >= thread_subgraph_indices.size()) {
        throw out_of_range("The thread ID has no assigned subgraph range");
    }

    const size_t start = thread_subgraph_indices[thread_id];
    const size_t end = thread_subgraph_indices[thread_id + 1];
    if (start > end || end > local_subgraph_paths.size()) {
        throw invalid_argument("Thread subgraph indices are invalid");
    }

    thread_subgraph_paths = span<const string>(local_subgraph_paths).subspan(start, end - start);
}

void readin_local_subgraphs(span<const string> thread_subgraph_paths,
                            vector<graph>& local_subgraphs,
                            httplib::Client& graph_file_client) {
    local_subgraphs.clear();
    local_subgraphs.reserve(thread_subgraph_paths.size());

    for (const string& subgraph_path : thread_subgraph_paths) {
        graph subgraph;
        subgraph.offsets = decode_indices(
            get_graph_file(graph_file_client, subgraph_path, "offsets.bin"), "offsets.bin");
        subgraph.neighbors = decode_indices(
            get_graph_file(graph_file_client, subgraph_path, "neighbors.bin"), "neighbors.bin");
        subgraph.embeddings = decode_embeddings(
            get_graph_file(graph_file_client, subgraph_path, "embeddings.bin"));
        subgraph.doc_indices = decode_indices(
            get_graph_file(graph_file_client, subgraph_path, "doc_indices.bin"),
            "doc_indices.bin");
        subgraph.chunk_byte_offsets = decode_indices(
            get_graph_file(graph_file_client, subgraph_path, "chunk_byte_offsets.bin"),
            "chunk_byte_offsets.bin");
        subgraph.chunk_byte_lengths = decode_indices(
            get_graph_file(graph_file_client, subgraph_path, "chunk_byte_lengths.bin"),
            "chunk_byte_lengths.bin");
        subgraph.doc_paths = nlohmann::json::parse(
            get_graph_file(graph_file_client, subgraph_path, "doc_paths.json"))
                                 .get<vector<string>>();

        validate_graph(subgraph);
        local_subgraphs.push_back(move(subgraph));
    }
}

void run_query_host(int rank, const string& graph_host_url, const string& source_host_url,
                    int query_port, const vector<string>& local_subgraph_paths) {
    const int requested_worker_count = static_cast<int>(local_subgraph_paths.size());
    const int requested_team_size = requested_worker_count + 1;

    vector<size_t> thread_subgraph_indices;
    local_query_gate gate;
    query_inbox inbox;
    httplib::Server server;
    thread server_thread;
    string startup_error;
    int worker_count = 0;
    bool startup_failed = false;

    omp_set_dynamic(0);
    #pragma omp parallel num_threads(requested_team_size) shared(worker_count, startup_failed, startup_error)
    {
        const int thread_id = omp_get_thread_num();

        #pragma omp single
        {
            worker_count = omp_get_num_threads() - 1;
            if (worker_count == 0 && !local_subgraph_paths.empty()) {
                startup_failed = true;
                startup_error = "OpenMP did not create a graph worker thread";
            } else if (worker_count > 0) {
                calc_thread_subgraph_indices(worker_count, local_subgraph_paths.size(),
                                             thread_subgraph_indices);
                gate.worker_results.resize(static_cast<size_t>(worker_count));
            }
        }

        vector<graph> local_subgraphs;
        span<const string> thread_subgraph_paths;
        if (thread_id > 0) {
            const int worker_id = thread_id - 1;
            try {
                extract_view_of_local_subgraph_paths(local_subgraph_paths,
                                                     thread_subgraph_indices,
                                                     worker_id,
                                                     thread_subgraph_paths);
                httplib::Client graph_file_client(graph_host_url.c_str());
                readin_local_subgraphs(thread_subgraph_paths, local_subgraphs,
                                       graph_file_client);
            } catch (const exception& error) {
                #pragma omp critical(arkg_startup_error)
                {
                    startup_failed = true;
                    if (startup_error.empty()) startup_error = error.what();
                }
            }
        }

        #pragma omp barrier

        if (thread_id == 0) {
            int local_failure = startup_failed ? 1 : 0;
            int any_failure = 0;
            MPI_Allreduce(&local_failure, &any_failure, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD);
            startup_failed = any_failure != 0;

            if (!startup_failed && rank == 0) {
                register_query_endpoint(server, inbox);
                if (!server.bind_to_port("127.0.0.1", query_port)) {
                    startup_failed = true;
                    startup_error = "Failed to bind the query server to port " +
                                    to_string(query_port);
                }
            }

            local_failure = startup_failed ? 1 : 0;
            MPI_Bcast(&local_failure, 1, MPI_INT, 0, MPI_COMM_WORLD);
            startup_failed = local_failure != 0;

            if (!startup_failed && rank == 0) {
                server_thread = thread([&server] { server.listen_after_bind(); });
            }
        }

        #pragma omp barrier

        if (!startup_failed) {
            if (thread_id == 0) {
                run_query_coordinator(rank, source_host_url, gate,
                                      rank == 0 ? &inbox : nullptr);
            } else {
                run_graph_worker(static_cast<size_t>(thread_id - 1), rank,
                                 thread_subgraph_paths, local_subgraphs, gate);
            }
        }
    }

    if (server_thread.joinable()) {
        server.stop();
        server_thread.join();
    }
    if (startup_failed) {
        if (startup_error.empty()) {
            startup_error = "A different rank failed while starting the query host";
        }
        throw runtime_error(startup_error);
    }
}
