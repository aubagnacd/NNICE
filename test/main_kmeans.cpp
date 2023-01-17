// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "clusterer.h"
#include <chrono>
#include <set>

int main(int argc, char** argv) {
    Kmeans clusterer;
    std::string centroids_file, inputs_file, norm_file;
    if (argc == 4) {
        centroids_file = argv[1];
        inputs_file = argv[2];
        norm_file = argv[3];
    } else {
        cout << "usage - mpirun -np 1 ./mainKmeans centroids_list_file inputs_file norm_file" << endl;
        cout << "centroid_list_file : filename of file containing list of centroids (string)" << endl;
        cout << "inputs_file : filename of file containing list of vectors to classify (string)" << endl;
        cout << "norm_file : filename of file containing normalization vector (string)" << endl;
        exit(1);
    }
    // Init clusterer
    clusterer.init(centroids_file);
    clusterer.read_input_norm(norm_file);
    clusterer.log_threshold = 1.0e-12;
    //
    // Read inputs
    //
    std::ifstream infile(inputs_file);
    int n_cases = 0;
    std::string line;
    while(getline(infile, line)) n_cases++;
    infile.clear();
    infile.seekg(0);
    double** test_pt = new double*[n_cases];
    for (size_t i = 0; i < n_cases; i++) {
        test_pt[i] = new double[clusterer.n_dims];
        for (size_t j = 0; j < clusterer.n_dims; j++) infile >> test_pt[i][j];
    }
    //
    // Check case reading (for debug purpose)
    //
    // for (size_t i = 0; i < n_cases; i++) {
    //     cout << "case :" << test_pt[i][0];
    //     for (size_t j = 1; j < clusterer.n_dims; j++) cout << " - " << test_pt[i][j];
    //     cout << endl;
    // }
    //
    // Clustering
    //
    int ind = 0;
    double* inputs = new double[clusterer.n_dims];
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n_cases; i++) {
        for (size_t k = 0; k < clusterer.n_dims; k++) inputs[k] = test_pt[i][k];
        clusterer.apply_log_transform(inputs);
        clusterer.normalize_input(inputs);
        ind = clusterer.run(inputs);
        cout << ind << endl;
    }
    auto end = std::chrono::steady_clock::now();
    cout << "time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << endl << endl;
    return 0;
}