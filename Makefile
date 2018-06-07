NVCC=nvcc
NVCCFLAGS=-std=c++14
BIN=for-comp.out

$(BIN): main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<
