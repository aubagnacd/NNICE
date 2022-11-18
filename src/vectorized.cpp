// Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "vectorized.h"

double* VectInference::run_ai(double* input_ai)
{
    throw std::logic_error("Vectorized inference need size of batch to be given");
}

double* VectInference::run_ai(double* input_ai, const size_t& size)
{
    if (size != neuronLayers[0].rows()) {
        for (size_t i = 0; i < neuronLayers.size(); i++) neuronLayers[i].resize(size,topology[i]);
    }
    MappedMat input_vector(input_ai, size, n_input_ai);
    propagateForward(input_vector);
    return neuronLayers.back().data();
}

void VectInference::propagateForward(MappedMat& input) {
    neuronLayers[0] = input;
    for (size_t i = 1; i < topology.size(); i++) {
        (*this.*layers[i-1])(i);
    }
}

void VectInference::AddLayer(size_t i) {
    neuronLayers.push_back(Matrix());
}

void VectInference::AddId() {
    activationFunctions.push_back(id);
}

void VectInference::AddReLU() {
    activationFunctions.push_back(reLU);
}

void VectInference::AddTanh() {
    activationFunctions.push_back(tanh);
}

void VectInference::AddSwish() {
    activationFunctions.push_back(swish);
}

void VectInference::AddDense() {
    layers.push_back(&VectInference::Dense);
}

void VectInference::AddResBlock2() {
    layers.push_back(&VectInference::ResBlock2);
}

void VectInference::Dense(size_t& i) {
    neuronLayers[i].noalias() = neuronLayers[i - 1] * weights[i - 1];
    neuronLayers[i].rowwise() += bias[i-1];
    activationFunctions[i-1](neuronLayers[i]);
}

void VectInference::ResBlock2(size_t& i) {
    neuronLayers[i] = neuronLayers[i - 2];
    neuronLayers[i].noalias() += neuronLayers[i - 1] * weights[i - 1];
    neuronLayers[i].rowwise() += bias[i-1];
    activationFunctions[i-1](neuronLayers[i]);
}