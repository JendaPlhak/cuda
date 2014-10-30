#include <cublas_v2.h>

#define pow_2(x) ( ((x) * (x)) )

__global__
void atoms_difference(sMolecule A, sMolecule B,
                        float * d_result,
                        int width,
                        int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i * width + j < size) {
        // printf("processing (%d, %d)\n", i, j);
        if (i < width && i < j && j < width) {
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
            d_result[i * width + j] = pow_2(da - db);
        } else {
            d_result[i * width + j] = 0.0f;
        }
    }
}

float * array_malloc_GPU(int size)
{
    float * d_array;
    cudaMalloc(&d_array, size * sizeof(float));
    return d_array;
}

float * array_to_GPU(float * array, int size)
{
    float * d_array = array_malloc_GPU(size);
    cudaMemcpy(d_array, array,
                size * sizeof(float),
                cudaMemcpyHostToDevice);
    return d_array;
}

sMolecule molecule_to_GPU(sMolecule A, int size)
{
    sMolecule d_A;
    d_A.x = array_to_GPU(A.x, size);
    d_A.y = array_to_GPU(A.y, size);
    d_A.z = array_to_GPU(A.z, size);
    return d_A;
}

void free_molecule(sMolecule d_A)
{
    cudaFree(d_A.x);
    cudaFree(d_A.y);
    cudaFree(d_A.z);
}

float solveGPU(sMolecule d_A, sMolecule d_B, int n) {

    int BLOCK_SIZE  = 8;
    int result_size = pow_2(n);

    float *d_result;

    cudaError err = cudaMalloc(&d_result, result_size * sizeof(float));
    if ( cudaSuccess != err ) {
        fprintf( stderr, "Cuda error in file '%s' in line %i : %s.\n",
                 __FILE__, __LINE__, cudaGetErrorString( err) );
        return 0.0f;
    }

    int GRID_SIZE = (n / BLOCK_SIZE) + 1;
    printf("Grid size: %d\n", GRID_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(GRID_SIZE, GRID_SIZE);

    atoms_difference<<<dimGrid, dimBlock>>>
                    (d_A, d_B, d_result, n, result_size);;

    cublasStatus_t ret;
    cublasHandle_t handle;
    ret = cublasCreate(&handle);

    float RMSD = 0.0f;
    // sum using cublas reduction algorithm
    cublasSasum(handle, result_size, d_result, 1, &RMSD);

    cudaFree(d_result);
	return sqrt(1 / ((float)n * ((float)n - 1)) * RMSD);
}
