#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <string>

#define pow_2(x) ( ((x) * (x)) )

// ####### BLOCK SIZE ######
#define BLOCK_SIZE_BIG_750 512
// #define BLOCK_SIZE_BIG_480 320

// #define BLOCK_SIZE_SMALL_750 96
// #define BLOCK_SIZE_SMALL_480 64

#define BLOCK_SIZE_BIG_480 10

#define BLOCK_SIZE_SMALL_750 10
#define BLOCK_SIZE_SMALL_480 10
// #########################

// ####### UNROLLING #######
#define UNROLL_N_BIG_750 32
// #define UNROLL_N_BIG_480 64

// #define UNROLL_N_BIG_750 1
#define UNROLL_N_BIG_480 1

// #define UNROLL_N_SMALL_750 16
// #define UNROLL_N_SMALL_480 32

#define UNROLL_N_SMALL_750 1
#define UNROLL_N_SMALL_480 1

// #########################

// ####### UNROLLING #######
#define INNER_N_ 2
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

struct Atom {
    float x, y, z;
};

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

template<unsigned N>
__device__ __host__
inline uint
getGridSum(int rows)
{
    int GRID_SIZE   = ((rows - (rows % N)) * (1 + rows / N)) / 2;
    GRID_SIZE      += (int) ceil(rows / (float) N) * (rows % N);
    return GRID_SIZE;
}

//lambda unroller
template<int Begin, int End, int Step = 1>
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

template<unsigned BLOCK_SIZE, unsigned UNROLL_N, unsigned INNER_N, bool diagonal_block, bool end_block, bool is_big>
__device__ inline
float loop(const int size, const int i, const int begin,
           const Atom (&a)[INNER_N], const Atom (&b)[INNER_N],
           const float A_x[BLOCK_SIZE], const float A_y[BLOCK_SIZE], const float A_z[BLOCK_SIZE],
           const float B_x[BLOCK_SIZE], const float B_y[BLOCK_SIZE], const float B_z[BLOCK_SIZE])
{
    float sum = 0.0;

    auto body = [&] (int j) {
        auto inner_loop = [&] (int k) {
            float diff_x = A_x[j] - a[k].x;
            float diff_y = A_y[j] - a[k].y;
            float diff_z = A_z[j] - a[k].z;

            float d_sumA = pow_2(diff_x) + pow_2(diff_y) + pow_2(diff_z);

            diff_x = B_x[j] - b[k].x;
            diff_y = B_y[j] - b[k].y;
            diff_z = B_z[j] - b[k].z;

            float d_sumB = pow_2(diff_x) + pow_2(diff_y) + pow_2(diff_z);

            sum += d_sumA + d_sumB;
            sum += -2.f * sqrt(d_sumA * d_sumB);
        };

        if (not is_big || not diagonal_block || i < begin + j) { // Real index of Atom corresponding to j.
            UnrollerL<0, INNER_N>::step(inner_loop, 0);
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

template <unsigned INNER_N>
__device__ inline
void getIndexes(const uint block_idx, int & _row, int & _col)
{
    int lower_row = (sqrt(8.f * block_idx * INNER_N + pow_2(INNER_N)) - INNER_N) / 2.f;
    for (int row = lower_row; row <= lower_row + INNER_N; ++row) {
        uint sum = getGridSum<INNER_N>(row);
        if (block_idx == sum) {
            _row = row;
            _col = block_idx - sum;
        }
    }
    _row = 0;
    _col = 0;
}

template <unsigned BLOCK_SIZE, unsigned UNROLL_N, unsigned INNER_N, bool is_big>
__global__
void atoms_difference(const sMolecule A, const sMolecule B,
                        float * d_result,
                        const int n)
{
    float sum = 0.f;

    __shared__ int row, col;
    __shared__ bool diagonal_block;
    __shared__ bool end_block;

    if (0 == threadIdx.x) {
        getIndexes<INNER_N>(threadIdx.x, row, col);

        diagonal_block = row == col * INNER_N;
    }

    __syncthreads();

    int block_begin = col * BLOCK_SIZE * INNER_N;
    int i     = block_begin + threadIdx.x * INNER_N;
    int begin = row * BLOCK_SIZE;

    __shared__ float A_x[BLOCK_SIZE], A_y[BLOCK_SIZE], A_z[BLOCK_SIZE];
    A_x[threadIdx.x] = A.x[begin + threadIdx.x];
    A_y[threadIdx.x] = A.y[begin + threadIdx.x];
    A_z[threadIdx.x] = A.z[begin + threadIdx.x];

    __shared__ float B_x[BLOCK_SIZE], B_y[BLOCK_SIZE], B_z[BLOCK_SIZE];
    B_x[threadIdx.x] = B.x[begin + threadIdx.x];
    B_y[threadIdx.x] = B.y[begin + threadIdx.x];
    B_z[threadIdx.x] = B.z[begin + threadIdx.x];

    Atom a[INNER_N];
    Atom b[INNER_N];

    if (i >= n) {
        goto REDUCTION;
    } else {
        auto body = [&] (int j) {
            a[j].x = A.x[i + j];
            a[j].y = A.y[i + j];
            a[j].z = A.z[i + j];

            b[j].x = B.x[i + j];
            b[j].y = B.y[i + j];
            b[j].z = B.z[i + j];
        };

        UnrollerL<0, INNER_N>::step(body, 0);

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
            sum = loop<BLOCK_SIZE, UNROLL_N, INNER_N,
                         true, true, is_big>
                                    (size, i, begin,
                                     a, b,
                                     A_x, A_y, A_z,
                                     B_x, B_y, B_z);
        } else if (true == diagonal_block && false == end_block) {
            sum = loop<BLOCK_SIZE, UNROLL_N, INNER_N,
                         true, false, is_big>
                                    (size, i, begin,
                                     a, b,
                                     A_x, A_y, A_z,
                                     B_x, B_y, B_z);
        } else if (false == diagonal_block && true == end_block) {
            sum = loop<BLOCK_SIZE, UNROLL_N, INNER_N,
                         false, true, is_big>
                                    (size, i, begin,
                                     a, b,
                                     A_x, A_y, A_z,
                                     B_x, B_y, B_z);
        } else {
            sum = loop<BLOCK_SIZE, UNROLL_N, INNER_N,
                         false, false, is_big>
                                    (size, i, begin,
                                      a, b,
                                      A_x, A_y, A_z,
                                      B_x, B_y, B_z);
        }
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
    for (int i = 0; i < divisible_2(BLOCK_SIZE); ++i) {
        size_red /= 2;
        if (threadIdx.x >= size_red) {
            return;
        } else {
            reduction[threadIdx.x] += reduction[size_red + threadIdx.x];
        }
        __syncthreads();
    }

    // __syncthreads();
    // while (size_red % 2 == 0) {
    //     size_red /= 2;
    //     if (threadIdx.x >= size_red) {
    //         return;
    //     } else {
    //         reduction[threadIdx.x] += reduction[size_red + threadIdx.x];
    //     }
    //     __syncthreads();
    // }

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

template <unsigned BLOCK_SIZE, unsigned UNROLL_N, unsigned INNER_N, bool is_big>
float solveGPU_templated(const sMolecule d_A, const sMolecule d_B, const int n) {

    int rows        = n / BLOCK_SIZE + (n % BLOCK_SIZE == 0 ? 0 : 1);
    int cols        = rows / INNER_N;

    int GRID_SIZE   = getGridSum<INNER_N>(rows);

    float *d_result = NULL;
    float RMSD      = 0;

    cudaMemcpyToSymbol(d_final_result, &RMSD, sizeof(RMSD));

    atoms_difference<BLOCK_SIZE, UNROLL_N, INNER_N, is_big>
        <<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_result, n);

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
            return solveGPU_templated<BLOCK_SIZE_BIG_750, UNROLL_N_BIG_750,
                                        INNER_N_, true>(d_A, d_B, n);
        } else {
            return solveGPU_templated<BLOCK_SIZE_BIG_480, UNROLL_N_BIG_480,
                                        INNER_N_, true>(d_A, d_B, n);
        }
    } else {
        if (GPU_t::GTX_750 == GPU_TYPE) {
            return solveGPU_templated<BLOCK_SIZE_SMALL_750, UNROLL_N_SMALL_750,
                                        INNER_N_, false>(d_A, d_B, n);
        } else {
            return solveGPU_templated<BLOCK_SIZE_SMALL_480, UNROLL_N_SMALL_480,
                                        INNER_N_, false>(d_A, d_B, n);
        }
    }
}