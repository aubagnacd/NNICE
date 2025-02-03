// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "nnice_single.h"

double* SnglInference::run_ai(double* input_ai)
{
    MappedArray input_vector(input_ai, n_input_ai);
    propagateForward(input_vector);
    return neuronLayers.back().data();
}

double* SnglInference::run_ai(double* input_ai, const size_t& size)
{
    // TBD : add warning (size ignored)
    MappedArray input_vector(input_ai, n_input_ai);
    propagateForward(input_vector);
    return neuronLayers.back().data();
}

void SnglInference::propagateForward(MappedArray& input) {
    neuronLayers[0] = input;
    for (size_t i = 1; i < topology.size(); i++) {
        (*this.*layers[i-1])(i);
    }
}

void SnglInference::AddLayer(size_t i) {
   neuronLayers.push_back(RowVector(i));
   cacheLayers.push_back(RowVector(i));
   deltas.push_back(RowVector(i));
}

void SnglInference::AddId() {
    activationFunctions.push_back(id);
}

void SnglInference::AddReLU() {
    activationFunctions.push_back(reLU);
}

void SnglInference::AddTanh() {
    activationFunctions.push_back(tanh);
}

void SnglInference::AddSwish() {
    activationFunctions.push_back(swish);
}

void SnglInference::AddSigmoid() {
    activationFunctions.push_back(sigmoid);
}

void SnglInference::AddIdDerivative() {
    activationFunctionsDerivative.push_back(idDerivative);
}

void SnglInference::AddReLUDerivative() {
    activationFunctionsDerivative.push_back(reLUDerivative);
}

void SnglInference::AddTanhDerivative() {
    activationFunctionsDerivative.push_back(tanhDerivative);
}

void SnglInference::AddSwishDerivative() {
    activationFunctionsDerivative.push_back(swishDerivative);
}

void SnglInference::AddSigmoidDerivative() {
    activationFunctionsDerivative.push_back(sigmoidDerivative);
}

void SnglInference::AddDense() {
   layers.push_back(&SnglInference::Dense);
}

void SnglInference::AddResBlock2() {
   layers.push_back(&SnglInference::ResBlock2);
}

void SnglInference::Dense(size_t& i) {
    neuronLayers[i] = neuronLayers[i - 1] * weights[i - 1] + bias[i-1];
    cacheLayers[i] = neuronLayers[i - 1] * weights[i - 1] + bias[i-1];
    activationFunctions[i-1](neuronLayers[i]);
}

void SnglInference::ResBlock2(size_t& i) {
    neuronLayers[i] = neuronLayers[i - 1] * weights[i - 1] + bias[i-1] + neuronLayers[i - 2];
    cacheLayers[i] = neuronLayers[i - 1] * weights[i - 1] + bias[i-1] + neuronLayers[i - 2];
    activationFunctions[i-1](neuronLayers[i]);
}

void SnglInference::read_input_norm(const string& filename) {
   double** norm_param_X;
   norm_param_X = new double*[2];
   norm_param_X[0] = new double[n_input_ai];
   norm_param_X[1] = new double[n_input_ai];
   //
   std::ifstream fX(filename);
   for (int i = 0; i < n_input_ai; i++) {
      for (int j = 0; j < 2; j++) fX >> norm_param_X[j][i];
      norm_param_X[1][i] = 1.0 /(sqrt(norm_param_X[1][i]));
   }
   //
   norm_param_X0 = Eigen::Map<NormArray>(norm_param_X[0],n_input_ai);
   norm_param_X1 = Eigen::Map<NormArray>(norm_param_X[1],n_input_ai);
   //
   delete[] norm_param_X;
}

void SnglInference::read_output_norm(const string& filename) {
   double ** norm_param_Y;
   norm_param_Y = new double*[2];
   norm_param_Y[0] = new double[n_output_ai];
   norm_param_Y[1] = new double[n_output_ai];
   //
   std::ifstream fY(filename);
   for (int i = 0; i < n_output_ai; i++) {
      for (int j = 0; j < 2; j++) fY >> norm_param_Y[j][i];
      norm_param_Y[1][i] = sqrt(norm_param_Y[1][i]);
   }
   //
   norm_param_Y0 = Eigen::Map<NormArray>(norm_param_Y[0],n_output_ai);
   norm_param_Y1 = Eigen::Map<NormArray>(norm_param_Y[1],n_output_ai);
   //
   delete[] norm_param_Y;
}

void SnglInference::normalize_input(double* state_X) {
   MappedArray input_vector(state_X,n_input_ai);
   input_vector = (input_vector - norm_param_X0)*norm_param_X1;
}

void SnglInference::normalize_output(double* state_Y) {
   MappedArray output_vector(state_Y,n_output_ai);
   output_vector = (output_vector - norm_param_Y0)/norm_param_Y1;
}

void SnglInference::denormalize_output(double* state_Y) {
   MappedArray output_vector(state_Y,n_output_ai);
   output_vector = (output_vector*norm_param_Y1) + norm_param_Y0;
}

void SnglInference::apply_log_transform(double* state_X) {
   double* ptr = state_X + 1;
   MappedArray input_vector(ptr,n_input_ai-1);
   input_vector = (input_vector < log_threshold).select(log_threshold, input_vector);
   input_vector = input_vector.log();
}

void SnglInference::inverse_log_transform(double* state_Y) {
   MappedArray input_vector(state_Y,n_output_ai);
   input_vector = input_vector.exp();
}

void SnglInference::apply_bct_transform(double* state_X) {
   double* ptr = state_X+1;
   MappedArray input_vector(ptr,n_input_ai-1);
   input_vector = (input_vector < 0.0).select(0.0, input_vector);
   input_vector = (input_vector.pow(bct_constant)-1.0)*rbct;
}

void SnglInference::inverse_bct_transform(double* state_Y) {
   MappedArray input_vector(state_Y,n_output_ai);
   input_vector = (bct_constant * input_vector + 1.0).pow(rbct);
}



void SnglInference::propagateBackward(MappedArray& input,MappedArray& output_expected_vector, double learning_rate)
{
    // calculate the errors made by neurons of last layer
    MappedArray output_vector(neuronLayers.back().data(),n_output_ai);

    RowVector output_error = error_function_derivative(output_vector,output_expected_vector);

    RowVector output_derivative = cacheLayers.back();
    activationFunctionsDerivative[0](output_derivative);

    RowVector output_delta = output_error.cwiseProduct(output_derivative);
    deltas[topology.size()-1] = output_delta;

    for (int i = topology.size()-2; i >= 0; i--) {
        Matrix map = Eigen::Map<Matrix>(weights[i].data(), topology[i], topology[i+1]);
        Matrix next_weights = map;
        RowVector next_delta = deltas[i+1];

        auto hidden_error =  next_weights * next_delta.transpose();
        RowVector hidden_derivative = cacheLayers[i];
        activationFunctionsDerivative[i](hidden_derivative);
        RowVector hidden_delta = hidden_error.transpose().cwiseProduct(hidden_derivative);
        deltas[i] = hidden_delta;
    }
    for (size_t i = 0; i < topology.size()-1; i++) {
        if (i == 0) {
            MappedArray  input_=input;
            update_weights(i, input_, learning_rate);
        }
        else {
            MappedArray  input_(neuronLayers[i].data(),neuronLayers[i].size());
            update_weights(i, input_, learning_rate);
        }
    }
}

void SnglInference::train(double* inputs,double* target_outputs,             
            double learning_rate, int epochs,
            size_t size) {

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        for (size_t i = 0; i < size; ++i) {
            MappedArray input(&inputs[i*n_input_ai],n_input_ai);
            MappedArray target_output(&target_outputs[i*n_output_ai],n_output_ai);

            propagateForward(input);
            MappedArray output(neuronLayers.back().data(),n_output_ai);
            double loss = error_function(output, target_output);
            epoch_loss += loss;

            propagateBackward(input, target_output, learning_rate);
        }
        epoch_loss /= size;
            
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << epoch_loss << std::endl;
    }
}