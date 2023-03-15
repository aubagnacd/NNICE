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

        void normalize_input(double*)
        void denormalize_output(double*)
        void read_input_norm(const string&)
        void read_output_norm(const string&)

        double* run_ai(double*)

        int n_input_ai
        int n_output_ai
        double log_threshold, bct_constant, rbct

cdef extern from "nnice_single.h":
    cdef cppclass SnglInference(Inference):
        SnglInference() except +

cdef extern from "clusterer.h":
    cdef cppclass Kmeans:
        Kmean() except +

        void init(const string&)
        int run(double*)
        
        void normalize_input(double*)
        void read_input_norm(const string&)

        double log_threshold, bct_constant, rbct
        int n_centroids, n_dims

