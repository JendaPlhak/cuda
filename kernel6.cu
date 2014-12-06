#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <algorithm>


#define pow_2(x) ( ((x) * (x)) )

#define BLOCK_SIZE_BIG   512
#define BLOCK_SIZE_SMALL 64

template<int BLOCK_SIZE, bool diagonal_block>
__host__ __device__ inline
float lopp(int size, int i, int begin,
           float a_x, float a_y, float a_z, float b_x, float b_y, float b_z,
           float A_x[BLOCK_SIZE], float A_y[BLOCK_SIZE], float A_z[BLOCK_SIZE],
           float B_x[BLOCK_SIZE], float B_y[BLOCK_SIZE], float B_z[BLOCK_SIZE])
{
    float sum = 0.0;
    #if BLOCK_SIZE > 128
        #pragma unroll 32
    #else
        #pragma unroll 64
    #endif
    for (int j = 0; j < size; ++j) {
        if (not diagonal_block || i < begin + j) { // Real index of Atom corresponding to j.
            // printf("processing (%d, %d)\n", i, index);
            float diff_x = A_x[j] - a_x;
            float diff_y = A_y[j] - a_y;
            float diff_z = A_z[j] - a_z;
            float da = sqrt(pow_2(diff_x) + pow_2(diff_y) + pow_2(diff_z));
            diff_x = B_x[j] - b_x;
            diff_y = B_y[j] - b_y;
            diff_z = B_z[j] - b_z;
            float db = sqrt(pow_2(diff_x) + pow_2(diff_y) + pow_2(diff_z));
            // printf("Ax diff [%f, %f, %f]\n",
            //             pow_2(A.x[i] - A.x[j]),
            //             pow_2(A.y[i] - A.y[j]),
            //             pow_2(A.z[i] - A.z[j]));
            // printf("Da: %f db: %f\n", da, db);
            // printf("saving result: %f\n", pow_2(da - db));
            sum += pow_2(da - db);
        }
    }
    return sum;
}

template <int BLOCK_SIZE>
__global__
void atoms_difference(sMolecule A, sMolecule B,
                        float * d_result,
                        int n,
                        int line_blocks)
{
    float a_x, a_y, a_z, b_x, b_y, b_z;
    __shared__ int row, col;
    __shared__ bool diagonal_block;
    if (0 == threadIdx.x) {
        // calculate current row by formula int(1/2 * (sqrt(8k + 1) - 1))
        row = (sqrt(8.0f * (float) blockIdx.x + 1) - 1) / 2.0f;
        col = blockIdx.x - (row * (row + 1)) / 2;

        diagonal_block = row == col;
    } 

    __syncthreads();

    int block_begin = col * BLOCK_SIZE;
    int i     = block_begin + threadIdx.x;
    int begin = row * BLOCK_SIZE;

    __shared__ float A_x[BLOCK_SIZE], A_y[BLOCK_SIZE], A_z[BLOCK_SIZE];
    A_x[threadIdx.x] = A.x[begin + threadIdx.x];
    A_y[threadIdx.x] = A.y[begin + threadIdx.x];
    A_z[threadIdx.x] = A.z[begin + threadIdx.x];

    __shared__ float B_x[BLOCK_SIZE], B_y[BLOCK_SIZE], B_z[BLOCK_SIZE];
    B_x[threadIdx.x] = B.x[begin + threadIdx.x];
    B_y[threadIdx.x] = B.y[begin + threadIdx.x];
    B_z[threadIdx.x] = B.z[begin + threadIdx.x];

    if (i >= n) {
        return;
    }

    // TODO: Does this provide any speedup?
    if (true == diagonal_block) {
        a_x = A_x[threadIdx.x];
        a_y = A_y[threadIdx.x];
        a_z = A_z[threadIdx.x];

        b_x = B_x[threadIdx.x];
        b_y = B_y[threadIdx.x];
        b_z = B_z[threadIdx.x];
    } else {
        a_x = A.x[i];
        a_y = A.y[i];
        a_z = A.z[i];

        b_x = B.x[i];
        b_y = B.y[i];
        b_z = B.z[i];
    }

    // calculate upper bound
    __shared__ int size;
    if (threadIdx.x == 0) {
        // calculate actual size of data block
        size = BLOCK_SIZE - fmax(0, (double) begin + BLOCK_SIZE - n);
    }
    __syncthreads();
    float sum;
    if (true == diagonal_block) {
        sum = lopp<BLOCK_SIZE, true>(size, i, begin,
                                       a_x, a_y, a_z, b_x, b_y, b_z,
                                       A_x, A_y, A_z,
                                       B_x, B_y, B_z);
    } else {
        sum = lopp<BLOCK_SIZE, false>(size, i, begin,
                                       a_x, a_y, a_z, b_x, b_y, b_z,
                                       A_x, A_y, A_z,
                                       B_x, B_y, B_z);
    }
    atomicAdd(d_result + i, sum);
}


float solveGPU(sMolecule d_A, sMolecule d_B, int n) {

    int BLOCK_SIZE;
    if (n > 2000) {
        BLOCK_SIZE = BLOCK_SIZE_BIG;
    } else {
        BLOCK_SIZE = BLOCK_SIZE_SMALL;
    }

    int line_blocks = n / BLOCK_SIZE + 1;
    int GRID_SIZE   = (line_blocks * (line_blocks + 1)) / 2;
    float *d_result;
    int result_size = n;

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

    if (n > 2000) {
        atoms_difference<BLOCK_SIZE_BIG><<<GRID_SIZE, BLOCK_SIZE>>>
                                            (d_A, d_B, d_result, n, line_blocks);
    } else {
        atoms_difference<BLOCK_SIZE_SMALL><<<GRID_SIZE, BLOCK_SIZE>>>
                                            (d_A, d_B, d_result, n, line_blocks);
    }

    float RMSD = 0;
    thrust::device_ptr<float> dptr(d_result);
    RMSD = thrust::reduce(thrust::device, dptr, dptr + result_size);

    cudaFree(d_result);
    return sqrt(1 / ((float)n * ((float)n - 1)) * RMSD);
}
