#include <cublas_v2.h>

#define pow_2(x) ( ((x) * (x)) )

__global__
void atoms_difference(sMolecule A, sMolecule B,
                        float * d_result,
                        int i,
                        int n)
{
    int j     = blockIdx.x * blockDim.x + threadIdx.x;
    int line  = threadIdx.y;
    int index = i + line;

    if (j < index && index < n) {
        // printf("processing (%d, %d)\n", i + line, j);
        float da = sqrt(pow_2(A.x[index] - A.x[j])
            + pow_2(A.y[index] - A.y[j])
            + pow_2(A.z[index] - A.z[j]));
        float db = sqrt(pow_2(B.x[index] - B.x[j])
            + pow_2(B.y[index] - B.y[j])
            + pow_2(B.z[index] - B.z[j]));
        // printf("Ax diff [%f, %f, %f]\n",
        //             pow_2(A.x[i] - A.x[j]),
        //             pow_2(A.y[i] - A.y[j]),
        //             pow_2(A.z[i] - A.z[j]));
        // printf("Da: %f db: %f\n", da, db);
        // printf("saving result: %f\n", pow_2(da - db));
        atomicAdd(d_result + j, pow_2(da - db));
    }
}

float solveGPU(sMolecule d_A, sMolecule d_B, int n) {

    int STEP         = 16;
    int BLOCK_SIZE_X = 64;
    int GRID_SIZE_X  = (n / BLOCK_SIZE_X) + 1;

    dim3 dimBlock(BLOCK_SIZE_X, STEP);
    dim3 dimGrid(GRID_SIZE_X, 1);

    float *d_result;
    int result_size = GRID_SIZE_X * BLOCK_SIZE_X;

    cudaError err = cudaMalloc(&d_result, result_size * sizeof(float));
    if ( cudaSuccess != err ) {
        fprintf( stderr, "Cuda error in file '%s' in line %i : %s.\n",
                 __FILE__, __LINE__, cudaGetErrorString(err) );
        return 0.0f;
    }
    err = cudaMemset(d_result, 0, result_size * sizeof(float));
    if ( cudaSuccess != err ) {
        fprintf( stderr, "Cuda error in file '%s' in line %i : %s.\n",
                 __FILE__, __LINE__, cudaGetErrorString(err) );
        return 0.0f;
    }

    for (int i = 0; i < n; i += STEP) {
        GRID_SIZE_X = (i / BLOCK_SIZE_X) + 1;
        atoms_difference<<<dimGrid, dimBlock>>>
                            (d_A, d_B, d_result, i, n);
    }

    cublasStatus_t ret;
    cublasHandle_t handle;
    ret = cublasCreate(&handle);

    float RMSD = 0.0f;
    // sum using cublas reduction algorithm
    cublasSasum(handle, result_size, d_result, 1, &RMSD);

    cudaFree(d_result);
    return sqrt(1 / ((float)n * ((float)n - 1)) * RMSD);
}