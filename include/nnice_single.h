// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef SNGL_INF_H
#define SNGL_INF_H
#include "inference.h"

class SnglInference : public Inference
{
    public:
        SnglInference() {};
        ~SnglInference() {};
        
        double* run_ai(double* input_ai);
        double* run_ai(double* input_ai, const size_t& size);

        void normalize_input(double* state_X);
        void denormalize_output(double* state_Y_norm);
        void read_input_norm(const string& filename);
        void read_output_norm(const string& filename);
        void select_log_species();
        void apply_log_transform(double* state_X);
        void inverse_log_transform(double* state_Y);
        void apply_bct_transform(double* state_X);
        void inverse_bct_transform(double* state_Y);

    protected:
        typedef void (*activ_ptr)(RowVector& input);
        typedef void (SnglInference::*layer_ptr)(size_t& i);

        void AddLayer(size_t i);
        void AddDense();
        void AddResBlock2();
        
        void AddReLU();
        void AddTanh();
        void AddId();
        void AddSwish();

        // activation functions should be static, no need to have 1 per object
        static void reLU(RowVector& input) {input = input.cwiseMax(0);};
        // tanh(x) = 1 - 2/(exp(2x)+1) : Eigen 3.4 only vectorizes tanh for float, the
        // double version falls back to scalar std::tanh (~6x slower). exp is vectorized
        // for double; max abs error vs std::tanh ~3e-16, and saturates cleanly to +-1.
        static void tanh(RowVector& input) {input = 1.0 - 2.0 / ((2.0 * input.array()).exp() + 1.0);};
        static void id(RowVector& input) {};
        static void swish(RowVector& input) {input = input.array() / (1.0 + Eigen::exp(-input.array()));};

        // Layer function pointer cannot be static (use of class attribute in function)
        void Dense(size_t& i);
        void ResBlock2(size_t& i);
        
        // function for forward propagation of data
        void propagateForward(MappedArray& input);

        friend class SnglInferencePerSpecies; // fused per-species path reads sub-net weights
      
        std::vector<RowVector, Eigen::aligned_allocator<RowVector>> neuronLayers; // stores the different layers of out network
        std::vector<activ_ptr> activationFunctions;
        std::vector<layer_ptr> layers;
};

#endif