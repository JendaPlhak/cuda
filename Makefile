CXX     := nvcc
CFLAGS  := -arch compute_20 -lcublas
CFLAGS  += -O3 --use_fast_math 

SRCS    := kernel_CPU.c
FRAMEWORKS := framework5 framework6 # framework1 framework2 framework3 framework4

all: $(FRAMEWORKS)

framework%: framework.cu kernel%.cu $(SRCS) Makefile
	$(CXX) $(CFLAGS) framework.cu -DKERNEL_NO_$* -o $@

clean:
	rm $(FRAMEWORKS)

.PHONY: clean all pall
