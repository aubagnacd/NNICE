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

void SnglInference::AddDense() {
   layers.push_back(&SnglInference::Dense);
}

void SnglInference::AddResBlock2() {
   layers.push_back(&SnglInference::ResBlock2);
}

void SnglInference::Dense(size_t& i) {
    neuronLayers[i] = neuronLayers[i - 1] * weights[i - 1] + bias[i-1];
    activationFunctions[i-1](neuronLayers[i]);
}

void SnglInference::ResBlock2(size_t& i) {
    neuronLayers[i] = neuronLayers[i - 1] * weights[i - 1] + bias[i-1] + neuronLayers[i - 2];
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

void SnglInference::denormalize_output(double* state_Y) {
   MappedArray input_vector(state_Y,n_output_ai);
   input_vector = (input_vector*norm_param_Y1) + norm_param_Y0;
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