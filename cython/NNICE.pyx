# Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
#-----------------------------------------------------------------------------
#cython: language_level=3

#######distutils: sources = [nnice_single.cpp, inference.cpp]

from NNICE cimport Inference,SnglInference
from libcpp.string cimport string
import numpy as np
cimport numpy as np

cdef class pyInference:
    cdef Inference *infptr

    def __cinit__(self):
        self.infptr = new Inference()

    def ImportNN(self, string model_path, string activation_path, string input_layer_name, string hidden_layers_prefix, string resblock_prefix, string output_layer_name):
        self.infptr.ImportNN(model_path, activation_path, input_layer_name, hidden_layers_prefix, resblock_prefix, output_layer_name)

cdef class pySnglInf(pyInference):
    cdef double* result

    def __cinit__(self):
        self.infptr = new SnglInference()

    def run_ai(self, np.ndarray[double, ndim=1] inputs_array):
         result = self.infptr.run_ai(&inputs_array[0])
         return np.asarray(<np.double_t[:self.infptr.n_output_ai]> result)