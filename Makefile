# NIRNAY — Linux/macOS build (CPU-only; the CUDA backend is built by build.bat on Windows).
#   make            -> bin/nirnay
#   make CUDA=1     -> also compile src/pdhg_cuda.cu with nvcc
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -fopenmp -Wall -Wno-sign-compare -Wno-unused-variable
SRC = src/model.cpp src/mps.cpp src/ldlt.cpp src/scaling.cpp src/presolve.cpp src/ipm.cpp \
      src/lu.cpp src/simplex.cpp src/mip.cpp src/pdhg.cpp src/solver.cpp src/main.cpp
OBJ = $(SRC:src/%.cpp=build/%.o)

ifeq ($(CUDA),1)
CXXFLAGS += -DNIRNAY_CUDA
OBJ += build/pdhg_cuda.o
LDLIBS += -lcudart
endif

bin/nirnay: $(OBJ) | bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

build/%.o: src/%.cpp src/*.hpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/pdhg_cuda.o: src/pdhg_cuda.cu src/pdhg.hpp | build
	nvcc -O3 -std=c++17 -DNIRNAY_CUDA -c $< -o $@

bin build:
	mkdir -p $@

clean:
	rm -rf build/*.o bin/nirnay

.PHONY: clean
