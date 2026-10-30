// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "clusterer.h"

Kmeans::~Kmeans()
{
}

void Kmeans::init(const string &model_path)
{
    // File layout: one line per input variable, one column per cluster.
    // Two formats exist:
    //   - legacy (pandas): first line is the cluster-index header "0 1 ... n-1"
    //   - ARF (np.savetxt): no header, data starts on the first line
    // The header is detected and skipped; it must NOT be skipped blindly,
    // otherwise the first variable (T) is lost and all rows shift by one.
    // (No std::vector here: keeps libNNICE.a linkable against older libstdc++.)
    std::ifstream fX(model_path);
    if (!fX) throw std::runtime_error("Kmeans::init: cannot open " + model_path);
    std::string line;
    double tmp;
    n_centroids = 0;
    n_dims = 0;
    bool has_header = false;
    //
    // Pass 1: count data lines and columns, detect header
    while(getline(fX, line)) {
        std::stringstream ss(line);
        int n_cols = 0;
        bool is_header = true;
        while( ss >> tmp ) {
            if (tmp != double(n_cols)) is_header = false;
            n_cols++;
        }
        if (n_cols == 0) {
            if (ss.eof()) continue; // blank line
            throw std::runtime_error("Kmeans::init: non-numeric line in " + model_path);
        }
        if (n_dims == 0 && !has_header && is_header) { has_header = true; n_centroids = n_cols; continue; }
        if (n_centroids == 0) n_centroids = n_cols;
        if (n_cols != n_centroids)
            throw std::runtime_error("Kmeans::init: inconsistent number of columns in " + model_path);
        n_dims++;
    }
    fX.clear();
    fX.seekg(0);
    //
    // Pass 2: read values
    centroids.resize(n_dims, n_centroids);
    if (has_header)
        for (int j = 0; j < n_centroids; j++) fX >> tmp;
    for (int i = 0; i < n_dims; i++)
        for (int j = 0; j < n_centroids; j++)
            fX >> centroids(i, j);
    // cout << centroids << endl; // for debug purpose
    // cout << "n_centroids = " << n_centroids << endl; // for debug purpose
    // cout << "n_dims = " << n_dims << endl; // for debug purpose
}

int Kmeans::run(double *input_kmeans)
{
    MappedVect input_vector(input_kmeans, n_dims);
    Matrix km = centroids;
    km.colwise() -= input_vector;
    Vector dists = km.colwise().squaredNorm();
    int index;
    double cf = dists.minCoeff(&index);
    return index;
}

void Kmeans::read_input_norm(const string &filename)
{
    double **norm_param_X;
    norm_param_X = new double *[2];
    norm_param_X[0] = new double[n_dims];
    norm_param_X[1] = new double[n_dims];
    //
    std::ifstream fX(filename);
    for (int i = 0; i < n_dims; i++)
    {
        for (int j = 0; j < 2; j++)
            fX >> norm_param_X[j][i];
        norm_param_X[1][i] = 1.0 / (sqrt(norm_param_X[1][i]));
    }
    //
    norm_param_X0 = Eigen::Map<NormArray>(norm_param_X[0], n_dims);
    norm_param_X1 = Eigen::Map<NormArray>(norm_param_X[1], n_dims);
    //
    // cout << norm_param_X0 << endl; // for debug purpose
    // cout << norm_param_X1 << endl; // for debug purpose
    //
    delete[] norm_param_X;
}

void Kmeans::normalize_input(double *state_X)
{
    MappedArray input_vector(state_X, n_dims);
    input_vector = (input_vector - norm_param_X0) * norm_param_X1;
}

void Kmeans::apply_log_transform(double *state_X)
{
    double *ptr = state_X + 1;
    MappedArray input_vector(ptr, n_dims - 1);
    input_vector = (input_vector < log_threshold).select(log_threshold, input_vector);
    input_vector = input_vector.log();
}

void Kmeans::apply_bct_transform(double *state_X)
{
    double *ptr = state_X + 1;
    MappedArray input_vector(ptr, n_dims - 1);
    input_vector = (input_vector < 0.0).select(0.0, input_vector);
    input_vector = (input_vector.pow(bct_constant) - 1.0) * rbct;
}