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
    std::ifstream fX(model_path);
    std::string line, item;
    n_centroids = 0;
    n_dims = 0;
    //
    while(getline(fX, line)) {
        n_dims++;
        //
        if (n_dims==1) {
            std::stringstream ss(line);
            while( ss >> item ) n_centroids++;
        }
    }
    n_dims--;
    fX.clear();
    fX.seekg(0);
    //
    centroids.resize(n_dims, n_centroids);
    //
    getline(fX, line); // skip first line
    double tmp;
    for (int i = 0; i < n_dims; i++)
    {
        for (int j = 0; j < n_centroids; j++) {
            fX >> tmp;
            centroids.row(i).col(j) << tmp;
        }
    }
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