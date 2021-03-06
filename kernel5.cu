#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>


#define pow_2(x) ( ((x) * (x)) )

#if MOLECULE_SIZE > 2000
    #define LARGE
    #define BLOCK_SIZE 512
#else
    #define SMALL
    #define BLOCK_SIZE 64
#endif

struct Atom {
    float x, y, z;
    Atom() {}
     __device__
    Atom(float x_, float y_, float z_) 
     : x(x_), y(y_), z(z_) {}
};

__global__
void atoms_difference(sMolecule A, sMolecule B,
                        float * d_result,
                        int n,
                        int line_blocks)
{
    float a_x, a_y, a_z, b_x, b_y, b_z;
    __shared__ int skip, quot, reminder;
    if (0 == threadIdx.x) {
        quot     = blockIdx.x / line_blocks;
        reminder = blockIdx.x % line_blocks;
        if (quot > reminder) {
            skip = 1;
        } else {
            skip = 0;
        }
    } 
    __syncthreads();
    if (skip == 1) {
        return;
    }

    int block_begin = (quot) * BLOCK_SIZE;
    int i     = block_begin + threadIdx.x;
    int begin = (reminder) * BLOCK_SIZE;

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

    __shared__ int copy_from_shared;
    if (0 == threadIdx.x) {
        if (block_begin == begin) {
            copy_from_shared = 1;
        } else {
            copy_from_shared = 0;
        }
    }
    __syncthreads();

    if (1 == copy_from_shared) {
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

    float sum = 0.0;
    #ifdef LARGE
        #pragma unroll 32
    #endif
    #ifdef SMALL
        #pragma unroll 64
    #endif
    for (int j = 0; j < BLOCK_SIZE; ++j) {
        int index = begin + j;
        if (index >= n) {
            break;
        }
        if (i < index) { 
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
    atomicAdd(d_result + i, sum);
}


float solveGPU(sMolecule d_A, sMolecule d_B, int n) {

    int line_blocks = n / BLOCK_SIZE + 1;
    int GRID_SIZE   = pow_2(line_blocks);
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

    atoms_difference<<<GRID_SIZE, BLOCK_SIZE>>>
                    (d_A, d_B, d_result, n, line_blocks);

    float RMSD = 0;
    thrust::device_ptr<float> dptr(d_result);
    RMSD = thrust::reduce(thrust::device, dptr, dptr + result_size);

    cudaFree(d_result);
    return sqrt(1 / ((float)n * ((float)n - 1)) * RMSD);
}
