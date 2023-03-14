# Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
#-----------------------------------------------------------------------------
# cython: language_level=3

from libcpp.string cimport string

import numpy as np
cimport numpy as np

cdef extern from "inference.h":
    cdef cppclass Inference:
        Inference() except +
        
        void ImportNN(const string&, const string&, const string&, const string&, const string&, const string&)

        double* run_ai(double*)

        int n_input_ai
        int n_output_ai
        double log_threshold, bct_constant, rbct

cdef extern from "nnice_single.h":
    cdef cppclass SnglInference(Inference):
        SnglInference() except +

