// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef INFERENCE_H
#define INFERENCE_H
#include "Eigen/Core"
#include "hdf5.h"
#include "rapidjson/document.h"
#include "vector"
#include <iostream>
#include <fstream>
#include <stdexcept>

using std::cout;
using std::endl;
using std::string;

class Inference
{
    public:
        Inference() {};
        ~Inference();
        
        void ImportNN(const string& model_path, const string& activation_path);
        void ImportNN(const string& model_path, const string& activation_path, const string& input_layer_name, const string& hidden_layers_prefix, const string& resblock_prefix, const string& output_layer_name);

        virtual double* run_ai(double* input_ai) {return nullptr;};
        virtual double* run_ai(double* input_ai, const size_t& size) {return nullptr;};

        virtual void normalize_input(double* state_X) {};
        virtual void denormalize_output(double* state_Y_norm) {};
        virtual void read_input_norm(const string& filename) {};
        virtual void read_output_norm(const string& filename) {};

        int n_input_ai;
        int n_output_ai;
        double log_threshold, bct_constant, rbct;

    protected:
        typedef Eigen::MatrixXd Matrix;
        typedef Eigen::RowVectorXd RowVector;
        typedef Eigen::Map<Eigen::ArrayXd> MappedArray;
        typedef Eigen::ArrayXd NormArray;

        void read_kernel_bias(const string& path);
        void read_activations(const string& path);

        // Layers names
        string input_layer, hidden_prefix, res2_prefix, output_layer;
        
        // HDF5 reader routine
        void add_bias_from_dataset(hid_t& h5_dset);
        void add_weight_from_dataset(hid_t& h5_dset);

        virtual void AddLayer(size_t i) {};
        virtual void AddDense() {};
        virtual void AddResBlock2() {};

        virtual void AddReLU() {};
        virtual void AddTanh() {};
        virtual void AddId() {};
        virtual void AddSwish() {};

        // Layer function pointer cannot be static (use of class attribute in function)
        virtual void Dense(size_t& i) {};
        virtual void ResBlock2(size_t& i) {};
        
        NormArray norm_param_X0, norm_param_X1, norm_param_Y0, norm_param_Y1;

        std::vector<size_t> topology;
        std::vector<Matrix, Eigen::aligned_allocator<Matrix>> weights; // the connection weights itself
        std::vector<RowVector, Eigen::aligned_allocator<RowVector>> bias; // stores the different layers of out network
};
#endif

