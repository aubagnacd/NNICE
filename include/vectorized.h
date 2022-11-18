// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef VECT_INF_H
#define VECT_INF_H
#include "inference.h"

using std::cout;
using std::endl;
using std::string;

class VectInference : public Inference
{
    public:
        VectInference() {};
        ~VectInference() {};

        double* run_ai(double* input_ai);
        double* run_ai(double* input_ai, const size_t& size);
 
    protected:
        typedef void (*activ_ptr)(Matrix& input);
        typedef void (VectInference::*layer_ptr)(size_t& i);
        typedef Eigen::Map<Eigen::MatrixXd> MappedMat;

        void AddLayer(size_t i);
        void AddDense();
        void AddResBlock2();
        
        void AddReLU();
        void AddTanh();
        void AddId();
        void AddSwish();

        // activation functions should be static, no need to have 1 per object
        static void reLU(Matrix& input) {input = input.cwiseMax(0);};
        static void tanh(Matrix& input) {input = input.array().tanh();};
        static void id(Matrix& input) {};
        static void swish(Matrix& input) {input = input.array() / (1.0 + Eigen::exp(-input.array()));};
        
        // Layer function pointer cannot be static (use of class attribute in function)
        void Dense(size_t& i);
        void ResBlock2(size_t& i);

        // function for forward propagation of data
        void propagateForward(MappedMat& input);

        std::vector<Matrix, Eigen::aligned_allocator<Matrix>> neuronLayers; // stores the different layers of out network
        std::vector<activ_ptr> activationFunctions;
        std::vector<layer_ptr> layers;
};

#endif