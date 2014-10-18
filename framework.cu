#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

// molecule is stored as cartesian coordinates of its atom, each dimmension
// is in separate array
struct sMolecule {
	float *x;
	float *y;
	float *z;
	// some data about atoms in real application, 
	// do not corrupt mem. access optimization here...
};

#include "kernel.cu"
#include "kernel_CPU.C"

#define N 10000

void createMolecules(sMolecule A, sMolecule B, int n) {
	for (int i = 0; i < n; i++) {
		// create atom in A at random position first
		A.x[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
		A.y[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
		A.z[i] = 1000.0f * (float)rand() / (float)RAND_MAX;
		// create atom in B near atom A
		// in small probability, create more displaced atom
		if ((float)rand() / (float)RAND_MAX < 0.01f) {
			B.x[i] = A.x[i] + 10.0f * (float)rand() / (float)RAND_MAX;
			B.y[i] = A.y[i] + 10.0f * (float)rand() / (float)RAND_MAX;
			B.z[i] = A.z[i] + 10.0f * (float)rand() / (float)RAND_MAX;
		}
		else {
			B.x[i] = A.x[i] + 1.0f * (float)rand() / (float)RAND_MAX;
                        B.y[i] = A.y[i] + 1.0f * (float)rand() / (float)RAND_MAX;
                        B.z[i] = A.z[i] + 1.0f * (float)rand() / (float)RAND_MAX;
		}
	}
}

int main(int argc, char **argv){
	sMolecule A, B;
	A.x = A.y = A.z = B.x = B.y = B.z = NULL;
	sMolecule dA, dB;
	dA.x = dA.y = dA.z = dB.x = dB.y = dB.z = NULL;
	float RMSD_CPU, RMSD_GPU;

	// parse command line
	int device = 0;
	if (argc == 2) 
		device = atoi(argv[1]);
	if (cudaSetDevice(device) != cudaSuccess){
		fprintf(stderr, "Cannot set CUDA device!\n");
		exit(1);
	}
	cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Using device %d: \"%s\"\n", device, deviceProp.name);

	// create events for timing
	cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

	// allocate and set host memory
	A.x = (float*)malloc(N*sizeof(A.x[0]));
	A.y = (float*)malloc(N*sizeof(A.y[0]));
	A.z = (float*)malloc(N*sizeof(A.z[0]));
	B.x = (float*)malloc(N*sizeof(B.x[0]));
        B.y = (float*)malloc(N*sizeof(B.y[0]));
        B.z = (float*)malloc(N*sizeof(B.z[0]));
	createMolecules(A, B, N);      
 
	// allocate and set device memory
	if (cudaMalloc((void**)&dA.x, N*sizeof(dA.x[0])) != cudaSuccess
	|| cudaMalloc((void**)&dA.y, N*sizeof(dA.y[0])) != cudaSuccess
	|| cudaMalloc((void**)&dA.z, N*sizeof(dA.z[0])) != cudaSuccess
	|| cudaMalloc((void**)&dB.x, N*sizeof(dB.x[0])) != cudaSuccess
        || cudaMalloc((void**)&dB.y, N*sizeof(dB.y[0])) != cudaSuccess
        || cudaMalloc((void**)&dB.z, N*sizeof(dB.z[0])) != cudaSuccess) {
		fprintf(stderr, "Device memory allocation error!\n");
		goto cleanup;
	}
	cudaMemcpy(dA.x, A.x, N*sizeof(dA.x[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(dA.y, A.y, N*sizeof(dA.y[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(dA.z, A.z, N*sizeof(dA.z[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(dB.x, B.x, N*sizeof(dB.x[0]), cudaMemcpyHostToDevice);
        cudaMemcpy(dB.y, B.y, N*sizeof(dB.y[0]), cudaMemcpyHostToDevice);
        cudaMemcpy(dB.z, B.z, N*sizeof(dB.z[0]), cudaMemcpyHostToDevice);

	// solve on CPU
        printf("Solving on CPU...\n");
	cudaEventRecord(start, 0);
	RMSD_CPU = solveCPU(A, B, N);
	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        printf("CPU performance: %f megapairs/s\n",
                float(N)*float(N-1)/2.0f/time/1e3f);

	// solve on GPU
	printf("Solving on GPU...\n");
	cudaEventRecord(start, 0);
	// run it 10x for more accurately timing results
        for (int i = 0; i < 10; i++)
		RMSD_GPU = solveGPU(dA, dB, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
	printf("GPU performance: %f megapairs/s\n",
                float(N)*float(N-1)/2.0f/time/1e2f);

	printf("CPU RMSD: %f\nGPU RMSD: %f\n", RMSD_CPU, RMSD_GPU);
	// check GPU results
	if ( fabsf((RMSD_CPU-RMSD_GPU) / ((RMSD_CPU+RMSD_GPU)/2)) < 0.01f)
		printf("Test OK :-).\n");
	else
		 fprintf(stderr, "Data mismatch: %f should be %f :-(\n", RMSD_GPU, RMSD_CPU);

cleanup:
	cudaEventDestroy(start);
        cudaEventDestroy(stop);

	if (dA.x) cudaFree(dA.x);
	if (dA.y) cudaFree(dA.y);
	if (dA.z) cudaFree(dA.z);
	if (dB.x) cudaFree(dB.x);
        if (dB.y) cudaFree(dB.y);
        if (dB.z) cudaFree(dB.z);
	if (A.x) free(A.x);
	if (A.y) free(A.y);
	if (A.z) free(A.z);
	if (B.x) free(B.x);
        if (B.y) free(B.y);
        if (B.z) free(B.z);

	return 0;
}

