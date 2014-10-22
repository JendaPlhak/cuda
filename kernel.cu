#define pow_2(x) ( (x * x) )


__global__
void atoms_difference(sMolecule A, sMolecule B,
                        float * d_result,
                        int n,
                        int result_length)
{
    int i = blockIdx.x * blockDim.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // if (i < n - 1 && j < n) {
    //     float da = sqrt(pow_2(A.x[i] - A.x[j])
    //             + pow_2(A.y[i] - A.y[j])
    //             + pow_2(A.z[i] - A.z[j]));
    //     float db = sqrt(pow_2(B.x[i] - B.x[j])
    //         + pow_2(B.y[i] - B.y[j])
    //         + pow_2(B.z[i] - B.z[j]));
    //     d_result[i * result_length + j] = pow_2(da - db);
    // } else {
        d_result[i * result_length + j] = 0.1f;
    // }
}

float solveGPU(sMolecule A, sMolecule B, int n) {

    int BLOCK_SIZE = 256;
    int GRID_SIZE  = n / BLOCK_SIZE + 1;

    int result_length = pow_2(GRID_SIZE * BLOCK_SIZE);

    float *result = (float*) malloc(result_length * sizeof(float));
    float *d_result;

    cudaMalloc(&d_result, result_length * sizeof(float));

    for (uint i = 0; i < result_length; ++i) {
        result[i] = 0.1;
    }

    cudaMemcpy(d_result, result,
                result_length * sizeof(float),
                cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(GRID_SIZE, GRID_SIZE);

    atoms_difference<<<(result_length + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>
                    (A, B, d_result, n, result_length);

    cudaMemcpy(result, d_result,
                result_length * sizeof(float),
                cudaMemcpyDeviceToHost);

    float RMSD = 0.0f;
    for (uint i = 0; i < result_length; ++i) {
        RMSD += result[i];
    }
	return RMSD;
}
