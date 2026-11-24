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

    if (argc != 3) {
        if (rank == 0) {
            std::fprintf(stderr, "Usage: %s <source_host_url> <graph_host_url>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    docs_container local_docs{};
    if (rank == 0) {
        httplib::Client client(argv[1]);
        docs_container global_docs{};

        vector<string> doc_paths;
        crawl_dir(client, doc_paths);

        populate_docs_container(client, global_docs, doc_paths);
        calc_offsets(global_docs, world_size);
        distribute_docs_to_ranks(global_docs, local_docs, world_size, rank);
    } else {
        receive_docs_from_master(local_docs, 0);
    }

    const int thread_count = omp_get_num_procs();

    // global trackers
    vector<float> embeddings;
    vector<size_t> chunk_sizes;
    vector<size_t> doc_indices;
    vector<size_t> chunk_byte_offsets;
    vector<size_t> chunk_byte_lengths;

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
        vector<size_t> local_doc_indices;
        vector<size_t> local_chunk_byte_offsets;
        vector<size_t> local_chunk_byte_lengths;
        chunk_memory(local_docs, thread_id, chunks, local_doc_indices,
                     local_chunk_byte_offsets, local_chunk_byte_lengths);
        chunk_sizes[thread_id] = chunks.size();

        // Allocate the shared embeddings and chunk metadata after all counts are ready.
        #pragma omp barrier
        #pragma omp single
        {
            resize_embeddings(embeddings, chunk_sizes, 384);
            resize_doc_indices(doc_indices, chunk_sizes);
            resize_chunk_byte_offsets(chunk_byte_offsets, chunk_sizes);
            resize_chunk_byte_lengths(chunk_byte_lengths, chunk_sizes);
        }
        #pragma omp barrier

        // Copy each thread's metadata into the same chunk order as its embeddings.
        copy_local_doc_indices_to_global(doc_indices, local_doc_indices, chunk_sizes, thread_id);
        copy_local_chunk_byte_offsets_to_global(chunk_byte_offsets, local_chunk_byte_offsets,
                                                chunk_sizes, thread_id);
        copy_local_chunk_byte_lengths_to_global(chunk_byte_lengths, local_chunk_byte_lengths,
                                                chunk_sizes, thread_id);

        // getting the thread-specific span of embeddings
        span<float> thread_embeddings;
        create_thread_embeddings_span(embeddings, chunk_sizes, 384, thread_id, thread_embeddings);

        // embedding the chunks
        embed_chunks(chunks, thread_embeddings);
    }

    // building adjacency matrix from embeddings
    constexpr float threshold = 0.8f;
    matrix_rep_adjacency adj;
    build_adjacency(adj, embeddings, threshold);

    // building CSR graph from adjacency matrix
    csr_graph csr;
    build_csr(csr, adj);

    // saving the graph to the file host
    try {
        const string build_id = save_graph(csr, embeddings, local_docs, doc_indices,
                                          chunk_byte_offsets, chunk_byte_lengths, argv[2], threshold);
        if (rank == 0) printf("Saved graph build %s\n", build_id.c_str());
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

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
