CXX     := nvcc
CFLAGS  := -arch compute_30 -lcublas
CFLAGS  += -O3 --use_fast_math 

SRCS    := kernel_CPU.c
FRAMEWORKS := framework6 # framework1 framework2 framework3 framework4 framework5 

all: $(FRAMEWORKS)

framework%: framework.cu kernel%.cu $(SRCS) Makefile
	$(CXX) $(CFLAGS) framework.cu -DKERNEL_NO_$* -o $@

clean:
	rm $(FRAMEWORKS)

.PHONY: clean all pall
