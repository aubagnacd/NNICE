all: main

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INC_DIR := $(ROOT_DIR)/../include

CXXFLAG := -DEIGEN_NO_DEBUG -march=native
INCLUDE := -I$(INC_DIR)
ifneq ($(origin EIGEN_INC),undefined)
  INCLUDE := -I$(EIGEN_INC) $(INCLUDE)
endif

ifneq ($(origin JSON_INC),undefined)
  INCLUDE := -I$(JSON_INC) $(INCLUDE)
endif

ifneq ($(origin HDF5_INC),undefined)
  INCLUDE := -I$(HDF5_INC) $(INCLUDE)
endif

LIB := -lhdf5 -lz
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
