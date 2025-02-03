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
        void normalize_output(double* state_Y);
        void denormalize_output(double* state_Y_norm);
        void read_input_norm(const string& filename);
        void read_output_norm(const string& filename);
        void select_log_species();
        void apply_log_transform(double* state_X);
        void inverse_log_transform(double* state_Y);
        void apply_bct_transform(double* state_X);
        void inverse_bct_transform(double* state_Y);
        RowVector error_function_derivative(const RowVector &output, const RowVector &target) {
            return output - target;
        }

        // function to calculate errors made by neurons in each layer

        double error_function(const RowVector &output, const RowVector &target) {
            return 0.5 * (output - target).squaredNorm();
        }

        // function to update the weights and bias by a given learning_rate

        void update_weights(size_t i, const RowVector& input, double learning_rate) {
            auto correction = learning_rate * input.transpose() * deltas[i+1];
            weights[i] -= correction;
            bias[i] -= learning_rate * deltas[i+1];
        }

        // function for backward propagation of errors made by neurons

        void propagateBackward(MappedArray& input,MappedArray& output_expected_vector, double learning_rate);

        // function to train the neural network give an array of data points

        void train(double* inputs, double* target_outputs, double learning_rate, int epochs, size_t size);

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
        void AddSigmoid();
        void AddReLUDerivative();
        void AddTanhDerivative();
        void AddIdDerivative();
        void AddSwishDerivative();
        void AddSigmoidDerivative();

        // activation functions should be static, no need to have 1 per object
        static void reLU(RowVector& input) {input = input.cwiseMax(0);};
        static void tanh(RowVector& input) {input = input.array().tanh();};
        static void id(RowVector& input) {};
        static void swish(RowVector& input) {input = input.array() / (1.0 + Eigen::exp(-input.array()));};
        static void sigmoid(RowVector& input) {input = 1.0/ (1.0 + Eigen::exp(-input.array()));};
        static void reLUDerivative(RowVector& input) {input = input.cwiseMax(0);};
        static void tanhDerivative(RowVector& input) {input = 1.0 - input.array().tanh()*input.array().tanh();};
        static void idDerivative(RowVector& input) {input.array() = 1.0;};
        static void swishDerivative(RowVector& input) {input = ( 1.0 + (1.0 + Eigen::exp(-input.array())) * ( 1.0 + input.array())) / ((1.0 + Eigen::exp(-input.array())) * (1.0 + Eigen::exp(-input.array())));};
        static void sigmoidDerivative(RowVector& input) {input = Eigen::exp(-input.array()) / ( (1.0 + Eigen::exp(-input.array())) * (1.0 + Eigen::exp(-input.array())) );};

        // Layer function pointer cannot be static (use of class attribute in function)
        void Dense(size_t& i);
        void ResBlock2(size_t& i);
        
        // function for forward propagation of data
        void propagateForward(MappedArray& input);
      
        std::vector<RowVector, Eigen::aligned_allocator<RowVector>> neuronLayers; // stores the different layers of out network
        std::vector<activ_ptr> activationFunctions;
        std::vector<layer_ptr> layers;

        std::vector<RowVector, Eigen::aligned_allocator<RowVector>> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers

        std::vector<activ_ptr> activationFunctionsDerivative;
        std::vector<RowVector, Eigen::aligned_allocator<RowVector>> deltas; // stores the error contribution of each neurons

        //void propagateBackward(RowVector& output); 
 
};

#endif