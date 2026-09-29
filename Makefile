# NIRNAY — Linux/macOS build.
#   make                 -> bin/nirnay            (dynamic, OpenMP)
#   make CUDA=1          -> also compile src/pdhg_cuda.cu with nvcc
#   make static          -> dist/nirnay-linux-x86_64  (fully static; use on Alpine/musl, see build.sh)
#   make static OMP=0    -> same, single-threaded (if the toolchain has no static libgomp)
CXX      ?= g++
OMP      ?= 1
NIRNAY_VERSION ?= 0.1.0
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unused-but-set-variable
CXXFLAGS += -DNIRNAY_VERSION=\"$(NIRNAY_VERSION)\"
ifeq ($(OMP),1)
CXXFLAGS += -fopenmp
endif
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

static: | dist
	$(CXX) $(CXXFLAGS) -static -s $(SRC) -o dist/nirnay-linux-x86_64

bin build dist:
	mkdir -p $@

clean:
	rm -rf build/*.o bin/nirnay dist

.PHONY: clean static
