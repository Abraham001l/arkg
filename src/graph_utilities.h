#pragma once
#include <cstddef>
#include <vector>
using namespace std;
constexpr int D = 384;

struct matrix_rep_adjacency {
    vector<vector<bool>> adjacency;
};

struct csr_graph {
    vector<size_t> neighbors;
    vector<size_t> offsets;
};

// strict cosine threshold; no self-edges or edges involving zero-norm vectors.
void build_adjacency(matrix_rep_adjacency &adj, vector<float> &embeddings, float threshold);

void build_csr(csr_graph &csr, const matrix_rep_adjacency &adj);

void norm(vector<float> &embeddings, vector<float> &norms);

void matrix_mul(vector<float> &embeddings, vector<float> &result, size_t d);
