# Copyright 2022-2023 IFPEN (www.ifpenergiesnouvelles.com)
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0
#-----------------------------------------------------------------------------
import os

from setuptools import setup, Extension
from Cython.Build import cythonize

nnice_ext = Extension(
        name="pyNNICE",
        sources=["NNICE.pyx"],
        language="c++", 
        extra_compile_args=['-fPIC'],
        extra_link_args=['-fPIC'],
        libraries=["hdf5", "NNICE"],
        library_dirs=[os.environ["HDF5_LIB"], "../lib/"],
        include_dirs=[os.environ["EIGEN_INC"],os.environ["JSON_INC"],os.environ["HDF5_INC"], "../include"]
      )

setup(
        name="pyNNICE",
        ext_modules=cythonize([nnice_ext])
    )