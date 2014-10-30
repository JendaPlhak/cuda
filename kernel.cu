#include <cublas_v2.h>

#define pow_2(x) ( ((x) * (x)) )

__global__
void atoms_difference(sMolecule A, sMolecule B,
                        float * d_result,
                        int width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

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

float solveGPU(sMolecule A, sMolecule B, int n) {

    int BLOCK_SIZE = 16;

    int result_size = pow_2(n);

    float *result = (float*) malloc(result_size * sizeof(float));
    float *d_result;

    cudaMalloc(&d_result, result_size * sizeof(float));

    // for (uint i = 0; i < result_size; ++i) {
    //     result[i] = 0.0f;
    // }

    // cudaMemcpy(d_result, result,
    //             result_size * sizeof(float),
    //             cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n / BLOCK_SIZE) + 1, (n / BLOCK_SIZE) + 1);

    atoms_difference<<<dimGrid, dimBlock>>>
                    (A, B, d_result, n);

    // cudaMemcpy(result, d_result,
    //             result_size * sizeof(float),
    //             cudaMemcpyDeviceToHost);


    cublasStatus_t ret;
    cublasHandle_t handle;
    ret = cublasCreate(&handle);

    float RMSD = 0.0f;
    // sum using cublas reduction algorithm
    cublasSasum(handle, result_size, d_result, 1, &RMSD);

    // for (uint i = 0; i < result_size; ++i) {
    //     // if (i % n == 0) {
    //     //     printf("\n%f ", result[i]);
    //     // } else {
    //     //     printf("%f ", result[i]);
    //     // }
    //     RMSD += result[i];
    // }
    // printf("\n");
	return sqrt(1 / ((float)n * ((float)n - 1)) * RMSD);
}
