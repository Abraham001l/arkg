#include "query_utilities.h"
#include <mpi.h>
#include <omp.h>
#include <httplib.h>
#include <charconv>
#include <cstdio>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv) {
    // initial MPI setup
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI_THREAD_FUNNELED is unavailable\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 5) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Usage: %s <graph_host_url> <source_host_url> <query_port> <build_id>\n",
                         argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    string graph_host_url = argv[1];
    string source_host_url = argv[2];

    int query_port;
    const string_view port_value(argv[3]);
    const auto [port_end, port_error] = from_chars(
        port_value.data(), port_value.data() + port_value.size(), query_port);
    if (port_error != errc{} || port_end != port_value.data() + port_value.size() ||
        query_port < 1 || query_port > 65535) {
        if (rank == 0) {
            std::fprintf(stderr, "query_port must be an integer between 1 and 65535\n");
        }
        MPI_Finalize();
        return 1;
    }

    const string build_id = argv[4];
    vector<string> global_subgraph_paths;
    int manifest_loaded = 1;
    if (rank == 0) {
        try {
            httplib::Client graph_file_client(graph_host_url.c_str());
            populate_subgraph_paths(graph_file_client, build_id, global_subgraph_paths);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Failed to load build '%s': %s\n", build_id.c_str(), error.what());
            manifest_loaded = 0;
        }
    }

    // All ranks must know whether to proceed before they wait for shard paths.
    MPI_Bcast(&manifest_loaded, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!manifest_loaded) {
        MPI_Finalize();
        return 1;
    }

    vector<string> local_subgraph_paths;
    // distribute and receive subgraph paths to/from ranks
    if (rank == 0) {
        // calculate indices for ranks to handle subgraphs
        vector<size_t> rank_subgraph_indices;
        calc_rank_subgraph_indices(world_size, global_subgraph_paths.size(),
                                    rank_subgraph_indices);

        // distribute the subgraph paths to each rank
        distribute_subgraph_paths_to_ranks(global_subgraph_paths, rank_subgraph_indices,
                                            local_subgraph_paths, rank);
    } else {
        // receive subgraph paths from the master rank
        receive_subgraph_paths_from_master(local_subgraph_paths, 0);
    }

    run_query_host(rank, graph_host_url, source_host_url, query_port,
                    local_subgraph_paths);

    MPI_Finalize();
    return 0;
}
