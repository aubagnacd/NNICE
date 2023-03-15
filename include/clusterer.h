// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef CLUSTERER_H
#define CLUSTERER_H
#include "Eigen/Core"
#include "vector"
#include <iostream>
#include <fstream>
#include <stdexcept>

using std::cout;
using std::endl;
using std::string;

class Kmeans
{
    public:
        Kmeans() {};
        ~Kmeans();
        
        void init(const string& model_path);
        int run(double* input_kmeans);
        
        void normalize_input(double* state_X);
        void read_input_norm(const string& filename);
        void apply_log_transform(double* state_X);
        void apply_bct_transform(double* state_X);

        double log_threshold, bct_constant, rbct;
        int n_centroids, n_dims;

    protected:
        typedef Eigen::MatrixXd Matrix;
        typedef Eigen::VectorXd Vector;
        typedef Eigen::Map<Eigen::VectorXd> MappedVect;
        typedef Eigen::Map<Eigen::ArrayXd> MappedArray;
        typedef Eigen::ArrayXd NormArray;

        NormArray norm_param_X0, norm_param_X1;
        Matrix centroids;
};
#endif