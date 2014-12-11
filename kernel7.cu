#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <string>

#define pow_2(x) ( ((x) * (x)) )

// ####### BLOCK SIZE ######
#define BLOCK_SIZE_BIG_750 512
#define BLOCK_SIZE_BIG_480 320

#define BLOCK_SIZE_SMALL_750 96
#define BLOCK_SIZE_SMALL_480 64
// #########################

// ####### UNROLLING #######
#define UNROLL_N_BIG_750 32
#define UNROLL_N_BIG_480 64

// #define UNROLL_N_BIG_750 1
// #define UNROLL_N_BIG_480 1

#define UNROLL_N_SMALL_750 16
#define UNROLL_N_SMALL_480 32
// #########################

enum GPU_t {
    NONE,
    GTX_750,
    GTX_480
};

GPU_t GPU_TYPE = NONE;

float CPU_reduction(float *d_data, const unsigned int n)
{
    float* h_odata = (float *) malloc(n * sizeof(float));
    cudaMemcpy(h_odata, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

    float result = 0.f;
    for (uint i = 0; i < n; i++) {
        result += h_odata[i];
    }
    free(h_odata);
    return result;
}

__device__ float d_final_result = 0.0f;

// expects d_data to be array of size n = 2^k
__global__ void 
GPU_reduction(float *d_data, unsigned int n)
{
    const unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
    
    d_final_result += d_data[index] + d_data[2 * index];
}

// For given n computes maximal number k which satisfies n % 2^k == 0 
__device__ constexpr int 
divisible_2(int n, int k = 0) {
    return n % 2 == 0 ? divisible_2(n / 2, k + 1) : k;
}

// For given n computes maximal number s such as n % s == 0 and s % 2 != 0
__device__ constexpr int 
factor_2(int n) {
    return n % 2 == 0 ? factor_2(n / 2) : n;
}

template<int Begin, int End, int Step = 1>
//lambda unroller
struct UnrollerL {
    template<typename Lambda>
    __device__ static void step(Lambda& func, const int offset) {
        func(Begin + offset);
        UnrollerL<Begin+Step, End, Step>::step(func, offset);
    }
};
//end of lambda unroller
template<int End, int Step>
struct UnrollerL<End, End, Step> {
    template<typename Lambda>
    __device__ static void step(Lambda& func, const int offset) {
    }
};

template<unsigned BLOCK_SIZE, unsigned UNROLL_N, bool diagonal_block, bool end_block, bool is_big>
__device__ inline
float loop(const int size, const int i, const int begin,
           const float a_x, const float a_y, const float a_z, const float b_x, const float b_y, const float b_z,
           const float A_x[BLOCK_SIZE], const float A_y[BLOCK_SIZE], const float A_z[BLOCK_SIZE],
           const float B_x[BLOCK_SIZE], const float B_y[BLOCK_SIZE], const float B_z[BLOCK_SIZE])
{
    float sum = 0.0;
    auto body = [&] (int j) {
        if (not is_big || not diagonal_block || i < begin + j) { // Real index of Atom corresponding to j.
            float diff_x = A_x[j] - a_x;
            float diff_y = A_y[j] - a_y;
            float diff_z = A_z[j] - a_z;

            float d_sumA = pow_2(diff_x) + pow_2(diff_y) + pow_2(diff_z);

            diff_x = B_x[j] - b_x;
            diff_y = B_y[j] - b_y;
            diff_z = B_z[j] - b_z;

            float d_sumB = pow_2(diff_x) + pow_2(diff_y) + pow_2(diff_z);

            sum += d_sumA + d_sumB;
            sum += -2.f * sqrt(d_sumA * d_sumB);
        }
    };

    if (end_block) {
        switch (BLOCK_SIZE) {
            case BLOCK_SIZE_BIG_750:
                #pragma unroll 64
                for (int j = 0; j < size; ++j) {
                    body(j);
                }
                break;
            case BLOCK_SIZE_BIG_480:
                #pragma unroll 32
                for (int j = 0; j < size; ++j) {
                    body(j);
                }
                break;
            default:
                for (int j = 0; j < size; ++j) {
                    body(j);
                }
        }
    } else {
        for (unsigned offset = 0; offset < BLOCK_SIZE; offset += UNROLL_N) {
            UnrollerL<0, UNROLL_N>::step(body, offset);
        }
    }
    if (not is_big && diagonal_block) {
        return sum / 2.f;
    } else {
        return sum;
    }
}

template <unsigned BLOCK_SIZE, unsigned UNROLL_N, bool is_big>
__global__
void atoms_difference(const sMolecule A, const sMolecule B,
                        float * d_result,
                        const int n,
                        const int line_blocks)
{
    float sum = 0.f;

    float a_x, a_y, a_z, b_x, b_y, b_z;
    __shared__ int row, col;
    __shared__ bool diagonal_block;
    __shared__ bool end_block;

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
        goto REDUCTION;
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
        int tmp_size = begin + BLOCK_SIZE - n;
        // calculate actual size of data block
        if (tmp_size < 0) {
            size = BLOCK_SIZE;
            end_block = false;
        } else {
            size = BLOCK_SIZE - tmp_size;
            end_block = true;
        }
    }
    __syncthreads();
    
    if (true == diagonal_block && true == end_block) {
        sum = loop<BLOCK_SIZE, UNROLL_N,
                     true, true, is_big>
                                (size, i, begin,
                                 a_x, a_y, a_z, b_x, b_y, b_z,
                                 A_x, A_y, A_z,
                                 B_x, B_y, B_z);
    } else if (true == diagonal_block && false == end_block) {
        sum = loop<BLOCK_SIZE, UNROLL_N,
                     true, false, is_big>
                                (size, i, begin,
                                 a_x, a_y, a_z, b_x, b_y, b_z,
                                 A_x, A_y, A_z,
                                 B_x, B_y, B_z);
    } else if (false == diagonal_block && true == end_block) {
        sum = loop<BLOCK_SIZE, UNROLL_N,
                     false, true, is_big>
                                (size, i, begin,
                                 a_x, a_y, a_z, b_x, b_y, b_z,
                                 A_x, A_y, A_z,
                                 B_x, B_y, B_z);
    } else {
        sum = loop<BLOCK_SIZE, UNROLL_N,
                     false, false, is_big>
                                (size, i, begin,
                                  a_x, a_y, a_z, b_x, b_y, b_z,
                                  A_x, A_y, A_z,
                                  B_x, B_y, B_z);
    }
    
REDUCTION:;
    __shared__ float reduction[BLOCK_SIZE];
    reduction[threadIdx.x] = sum;
    int size_red = BLOCK_SIZE;

    // auto body_reduction = [&] (int i) {
    //     int size = BLOCK_SIZE / (2 << i);
    //     if (threadIdx.x >= size) {
    //         return;
    //     } else {
    //         reduction[threadIdx.x] += reduction[size + threadIdx.x];
    //     }
    //     __syncthreads();
    // };
    // __syncthreads();
    // UnrollerL<0, divisible_2(BLOCK_SIZE)>::step(body_reduction, 0);

    __syncthreads();
    while (size_red % 2 == 0) {
        size_red /= 2;
        if (threadIdx.x >= size_red) {
            return;
        } else {
            reduction[threadIdx.x] += reduction[size_red + threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum = 0;
        auto body_add = [&] (int i) { 
            sum += reduction[i];
        };
        UnrollerL<0, factor_2(BLOCK_SIZE)>::step(body_add, 0);
        atomicAdd(&d_final_result, sum);
        }
}

constexpr bool 
isBig(const int n) {
    return n > 2000;
}

template <unsigned BLOCK_SIZE, unsigned UNROLL_N, bool is_big>
float solveGPU_templated(const sMolecule d_A, const sMolecule d_B, const int n) {

    int line_blocks = n / BLOCK_SIZE + (n % BLOCK_SIZE == 0 ? 0 : 1);
    int GRID_SIZE   = (line_blocks * (line_blocks + 1)) / 2;
    float *d_result = NULL;
    float RMSD      = 0;


    cudaMemcpyToSymbol(d_final_result, &RMSD, sizeof(RMSD));

    atoms_difference<BLOCK_SIZE, UNROLL_N, is_big><<<GRID_SIZE, BLOCK_SIZE>>>
                                            (d_A, d_B, d_result, n, line_blocks);
    
    cudaMemcpyFromSymbol(&RMSD, d_final_result, sizeof(RMSD));

    return sqrt(1 / ((float)n * ((float)n - 1)) * RMSD);
}

GPU_t getCurrentGPU() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    if ("GeForce GTX 750" == std::string(deviceProp.name)) {
        return GPU_t::GTX_750;
    } else {
        return GPU_t::GTX_480;
    }
}

float solveGPU(const sMolecule d_A, const sMolecule d_B, const int n) {

    if (NONE == GPU_TYPE) {
        GPU_TYPE = getCurrentGPU();
    }
    if (isBig(n)) {
        if (GPU_t::GTX_750 == GPU_TYPE) {
            return solveGPU_templated<BLOCK_SIZE_BIG_750,
                                      UNROLL_N_BIG_750, true>(d_A, d_B, n);
        } else {
            return solveGPU_templated<BLOCK_SIZE_BIG_480,
                                      UNROLL_N_BIG_480, true>(d_A, d_B, n);
        }
    } else {
        if (GPU_t::GTX_750 == GPU_TYPE) {
            return solveGPU_templated<BLOCK_SIZE_SMALL_750,
                                      UNROLL_N_SMALL_750, false>(d_A, d_B, n);
        } else {
            return solveGPU_templated<BLOCK_SIZE_SMALL_480,
                                      UNROLL_N_SMALL_480, false>(d_A, d_B, n);
        }
    }
}