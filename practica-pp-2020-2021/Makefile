# Common binaries
GCC             ?= g++
NVCC             ?= /usr/local/cuda/bin/nvcc

# Extra user f
EXTRA_NVCCFLAGS ?= -w 
EXTRA_LDFLAGS   ?= -w 

# OS-specific build flags
LDFLAGS   := -fopenmp -lcudart
CCFLAGS   += -O3 -m64
INCLUDES = -I/usr/local/cuda/include
LDNVCC = -L/usr/local/cuda/lib64
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM35) 

# Target rules
all: build

build: energy

solver_omp.o: solver_omp.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c -w $< -fopenmp
	
wtime.o: wtime.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c -w $< -fopenmp

energy_common.o: energy_common.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c -w $< -fopenmp

solver.o: solver.cu
	$(NVCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) $(GENCODE_FLAGS) -o $@ -c -w $< 
	
energy.o: energy.cpp
	$(GCC) $(CCFLAGS) $(EXTRA_CCFLAGS) $(INCLUDES) -o $@ -c -w $< -fopenmp

energy: solver_omp.o solver.o wtime.o energy_common.o energy.o 
	$(GCC) $(INCLUDES) $(CCFLAGS) $(LDNVCC) -o $@ $+ $(LDFLAGS) $(EXTRA_LDFLAGS)

run: build
	./energy

clean:
	rm -f energy *.o log

data:
	rm -f 2bsm* log *_cristalografico* 
