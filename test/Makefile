all: main

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INC_DIR := $(ROOT_DIR)/../include

CXXFLAG := -DEIGEN_NO_DEBUG -march=native
INCLUDE := -I$(HDF5_INC) -I$(EIGEN_INC) -I$(JSON_INC) -I$(INC_DIR)
LIB := -lhdf5
LFLAG := -L$(HDF5_LIB)

main:
	$(CXX) -O3 $(CXXFLAG) $(INCLUDE) -c ../src/inference.cpp
	$(CXX) -O3 $(CXXFLAG) $(INCLUDE) -c ../src/vectorized.cpp
	$(CXX) -O3 $(CXXFLAG) $(INCLUDE) -c ../src/nnice_single.cpp
	$(CXX) -O3 $(CXXFLAG) $(INCLUDE) $(LFLAG) -o mainProgram main.cpp inference.o vectorized.o nnice_single.o $(LIB)

debug:
	$(CXX) -g $(CXXFLAG) $(INCLUDE) -c ../src/inference.cpp
	$(CXX) -g $(CXXFLAG) $(INCLUDE) -c ../src/vectorized.cpp
	$(CXX) -O3 $(CXXFLAG) $(INCLUDE) -c ../src/nnice_single.cpp
	$(CXX) -g $(CXXFLAG) $(INCLUDE) $(LFLAG) -o mainProgram main.cpp inference.o vectorized.o nnice_single.o $(LIB)

clean:
	rm -f mainProgram *.o
