#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>


#define pow_2(x) ( ((x) * (x)) )

__global__
void atoms_difference(sMolecule A, sMolecule B,
                        float * d_result,
                        int width)
{
    // __shared__ int skip;
    // if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
    //     int i = blockIdx.x * blockDim.x + threadIdx.x;
    //     int j = blockIdx.y * blockDim.y + threadIdx.y;
    //     if (i < j) {
    //         skip = 0;
    //     } else {
    //         skip = 1;
    //     }
    // }
    // __syncthreads();
    // if (1 == skip) {
    //     return;
    // }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("processing (%d, %d)\n", i, j);
    if (i < j && j < width && i < width) {
        float da = sqrt(pow_2(A.x[i] - A.x[j])
            + pow_2(A.y[i] - A.y[j])
            + pow_2(A.z[i] - A.z[j]));
        float db = sqrt(pow_2(B.x[i] - B.x[j])
            + pow_2(B.y[i] - B.y[j])
            + pow_2(B.z[i] - B.z[j]));
        // printf("Ax diff [%f, %f, %f]\n",
        //             pow_2(A.x[i] - A.x[j]),
        //             pow_2(A.y[i] - A.y[j]),
        //             pow_2(A.z[i] - A.z[j]));
        // printf("Da: %f db: %f\n", da, db);
        // printf("saving result: %f\n", pow_2(da - db));
        atomicAdd(d_result + i, pow_2(da - db));
        // d_result[j] = pow_2(da - db);
    }
}


float solveGPU(sMolecule d_A, sMolecule d_B, int n) {

    int BLOCK_SIZE_X = 32;
    int BLOCK_SIZE_Y = 8;
    int GRID_SIZE_X  = (n / BLOCK_SIZE_X) + 1;
    int GRID_SIZE_Y  = (n / BLOCK_SIZE_Y) + 1;

    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid(GRID_SIZE_X, GRID_SIZE_Y);

    float *d_result;
    int result_size = (GRID_SIZE_X * BLOCK_SIZE_X);

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

    atoms_difference<<<dimGrid, dimBlock>>>
                    (d_A, d_B, d_result, n);

    float RMSD = 0;
    thrust::device_ptr<float> dptr(d_result);
    RMSD = thrust::reduce(thrust::device, dptr, dptr + result_size);

    cudaFree(d_result);
    return sqrt(1 / ((float)n * ((float)n - 1)) * RMSD);
}
