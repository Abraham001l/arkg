#include <cstdio>
#include <mpi.h>
#include "doc_utilities.h"
#include "embedding_utilities.h"
#include "graph_utilities.h"
#include <omp.h>
#include <httplib.h>
using namespace std;

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

    // checking if recieved all arguements
    if (argc != 3) {
        if (rank == 0) {
            std::fprintf(stderr, "Usage: %s <file_host_url> <directory>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    docs_container local_docs{};
    if (rank == 0) {
        httplib::Client client(argv[1]);
        docs_container global_docs{};

        vector<string> doc_paths;
        crawl_dir(client, argv[2], doc_paths);

        populate_docs_container(client, global_docs, doc_paths);
        calc_offsets(global_docs, world_size);
        distribute_docs_to_ranks(global_docs, local_docs, world_size, rank);
    } else {
        receive_docs_from_master(local_docs, 0);
    }

    const int thread_count = omp_get_num_procs();

    // creating embeddings vector for all threads
    vector<float> embeddings;

    // creating chunks global tracker for all threads
    vector<size_t> chunk_sizes;

    // starting the parallel chunking and embedding process
    #pragma omp parallel num_threads(thread_count)
    {
        #pragma omp single
        {
            const int team_size = omp_get_num_threads();
            calc_offsets(local_docs, team_size);
            chunk_sizes.resize(team_size);
        }

        const int thread_id = omp_get_thread_num();

        vector<string_view> chunks;
        chunk_memory(local_docs, thread_id, chunks);
        chunk_sizes[thread_id] = chunks.size();

        // creating large embedding vector for all threads
        #pragma omp barrier
        #pragma omp single
        {
            resize_embeddings(embeddings, chunk_sizes, 384);
        }
        #pragma omp barrier

        // getting the thread-specific span of embeddings
        span<float> thread_embeddings;
        create_thread_embeddings_span(embeddings, chunk_sizes, 384, thread_id, thread_embeddings);

        // embedding the chunks
        embed_chunks(chunks, thread_embeddings);
    }

    // building adjacency matrix from embeddings
    matrix_rep_adjacency adj;
    build_adjacency(adj, embeddings, 0.8f); // Initial threshold; tune for your documents.

    // building CSR graph from adjacency matrix
    csr_graph csr;
    build_csr(csr, adj);

    // printing the CSR graph for verification
    for (size_t i = 0; i < csr.offsets.size() - 1; ++i) {
        printf("Node %zu: ", i);
        for (size_t j = csr.offsets[i]; j < csr.offsets[i + 1]; ++j) {
            printf("%zu ", csr.neighbors[j]);
        }
        printf("\n");
    }

    MPI_Finalize();
}
