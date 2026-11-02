#include <cstdio>
#include <mpi.h>
#include "doc_utilities.h"
#include "embedding_utilities.h"
#include "graph_utilities.h"
using namespace std;

constexpr size_t one_GB = 1024 * 1024 * 1024;

int main(int argc, char** argv) {
    // initial MPI setup
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // picking master rank for doc retrieval
    int master_rank;
    bool doc_retriever_found = pick_doc_retriever("test_text_files", rank, master_rank);

    if (!doc_retriever_found) {
        printf("No doc retriever found.\n");
        MPI_Finalize();
        return 1;
    }

    // creating work container for each rank
    work_container w = {0};

    // having the master rank create 1GB of shared memory to read in docs
    if (rank == master_rank) {
        // allocating mem to read docs into
        auto mem = make_unique_for_overwrite<char[]>(one_GB);
        char* base_ptr = mem.get();

        // skeleton struct for docs
        docs_container d = {sizeof(size_t)*3, 0, 0};

        // reading docs into mem
        vector<string> doc_paths = {"test_text_files/doc1.txt", "test_text_files/doc2.txt", "test_text_files/doc3.txt", "test_text_files/doc4.txt",
        "test_text_files/doc5.txt", "test_text_files/doc6.txt", "test_text_files/doc7.txt", "test_text_files/doc8.txt", "test_text_files/doc9.txt",
        "test_text_files/doc10.txt"};

        populate_docs_container(d, base_ptr, doc_paths);
        calc_rank_offsets(d, base_ptr, world_size);

        // distributing docs to other ranks
        distribute_docs_to_ranks(d, base_ptr, world_size, rank);

        // filling work container for master rank
        fill_work_container_for_master_rank(w, d, base_ptr, rank);

        // freeing the docs container
        free_docs_container(d);
    } else {
        // receiving docs from master rank
        receive_docs_from_master(w, master_rank);
    }

    // chunking the work memory into lines
    std::vector<std::string_view> chunks;
    chunk_memory(w, chunks);

    // embedding the chunks
    vector<float> embeddings;
    embed_chunks(chunks, embeddings);

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
