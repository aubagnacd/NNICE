NNICE
=======

The project *NNICE* (Neural Network Inference in C made Easy) is a simple C++ library that aims at providing neural network inference capabilities without having to go through the cumbersome compilation of usual Machine Learning libraries C++ APIs. It has initially been developed for direct integration in 3D CFD codes. In these codes, the calls to NN often occur in a constrained environment in term of parallelization. The recipient code being already massively parallelized (usually), the inference method got to be efficient for numerous calls (vectorization is not always possible) in single-threaded CPU conditions (call often occurs in a distributed part of the code).

Note:
```diff
- Vectorized calls implementation is currently under development
```

# Current functionalities
So far, *NNICE* only reads MLP and ResNet with dense layers. The input format is also specific but it is described further in the *NN nomenclature* section and can be obtained natively from TensorFlow.

# Requirements
*NNICE* relies heavily on [EIGEN](https://gitlab.com/libeigen/eigen/-/releases/3.4.0) (3.4.0) for the matrix product and element-wise activation functions. Its input file parser relies on [HDF5](https://github.com/HDFGroup/hdf5) (1.10.5) and [Rapidjson](https://github.com/Tencent/rapidjson) (1.1.0). Versions indicated are the one tested and therefore preconized.

# Compilation
*NNICE* only requires five environment variables to be set:
```
CXX : C++ compiler
HDF5_INC : path to HDF5 headers
HDF5_LIB : path to compiled HDF5 library files
EIGEN_INC : path to EIGEN headers
JSON_INC : path to rapidjson heads
```

A simple Makefile is provided. *make* will compile the code, generate a test program in the *test* directory and the static library file to link against in the *lib* directory.

# Test
If the compilation ends without errors, you'll have an executable called *mainProgram* in the *test* directory. Its usage is indicated if you execute it without argument. Expected results can be found in *test/test_results.dat*. NNs are stored in *NNs" directory and input data for each case are hard-coded in *test/data.h* file.

You can also modify this test program to check if the NN you generated can be read by *NNICE*. It is highly recommended to do so before running simulations with *NNICE* in your code.

# NN nomenclature
Example of NN read by *NNICE* are given in *test/NNs* directory. The layers' names looked for by *NNICE* are user-defined with specific names for input and output layers and prefixes for dense MLP and ResNet layers. After the prefix, hidden layers should be numbered. Structure should:
+ "Input_layer_name"
  + "Input_layer_name"
    + kernel:0
    + bias:0
+ "hidden_layer_prefix1"
  + "hidden_layer_prefix1"
    + kernel:0
    + bias:0
+ "resblock_layer_prefix2"
  + "resblock_layer_prefix2"
    + kernel:0
    + bias:0
+ "Output_layer_name"
  + "Output_layer_name"
    + kernel:0
    + bias:0

Other layers will be ignored by *NNICE*

For activation functions, 4 functions are currently available:
|Functions|Keyword|
|Identity|id|
|ReLU|relu|
|tanh|tanh|
|swish|swish|
These keywords are taken from the respective functions in TensorFlow.

# Implementation in code
*include* and *lib* directories are to be taken directly from the git root (no install to prefixed path has been made yet). Once you got them in your code compilation environment, just copy the instantiation and call syntax you can find in the test folder or on the following:

+ Declare object
```cpp
Inference* my_nn;
```
+ Instantiate and import NN (same import method for both single and vectorized inference)
```cpp
my_nn = new SnglInference(); # single inference
my_vect_nn = new VectInference(); # vectorized inference

my_nn->ImportNN("path/to_model/model.h5","path/to_model/architecture.json");
# or
my_nn->ImportNN("path/to_model/model.h5","path/to_model/architecture.json", "input_layer_name", "hidden_layer_prefix", "resblock_layer_prefix", "output_layer_name");
```
+ Inference method
```cpp
result_ptr = my_nn->run_ai(input_ptr); # single inference
vresult_ptr = my_vect_nn->run_ai(input_ptr, batch_size); # vectorized inference
```
