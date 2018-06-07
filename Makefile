NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_60
BIN=for-comp.out

$(BIN): main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -rf $(BIN)
