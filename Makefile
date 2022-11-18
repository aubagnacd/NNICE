all: main

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INC_DIR := $(ROOT_DIR)/include

CXXFLAG := -DEIGEN_NO_DEBUG -march=native
INCLUDE := -I$(HDF5_INC) -I$(EIGEN_INC) -I$(JSON_INC) -I$(INC_DIR)
LIB := -lhdf5
LFLAG := -L$(HDF5_LIB)

main:
	+$(MAKE) -C test
	+$(MAKE) -C lib

debug:
	+$(MAKE) -C test debug
	+$(MAKE) -C lib debug

clean:
	+$(MAKE) -C test clean
	+$(MAKE) -C lib clean