#include <iostream>
#include <cuda_runtime.h>

// need to parameterize 3 values
__global__ void ElementWiseMultiply(float* A, float* B, float* C, int rows, int cols) 
{
    // compute the row and columns of threads using 2D indexing
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // bounds check
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] * B[idx];
    }
}
int main(void) 
{
    // define the matrix dimensions
    int rows = 3;
    int cols = 4;
    // calculate size or total # of bytes required
    // to store the matrix
    int size = rows * cols * sizeof(float);

    // 2. Allocate and intialize host memory for A, B, and C
    // allocate using malloc
    // cast to  the data type
    // and we allocate size as the number of bytes
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // initialize A and B by looping
    // filling host memory allocation 
    // with values
    for (int i = 0; i < rows * cols; i++) {
        // type matching to use f with the literal
        h_A[i] = 1.0f * i;
        h_B[i] = 2.0f * i;
    }

    // 3. Allocate vectors in device memory
    // Do this using cudamalloc (linear memory)
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // 4. Copy inputs from host to device using cudaMemcpy
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 5. Configure block and grid dimensions using blockDim
    // and gridDim
    // launch enough threads to cover all elements
    dim3 blockDim(16, 16);
    // perform ceiling division for the grid
    dim3 gridDim((cols + 15) / 16, (rows + 15) / 16);

    // 6. Launch the kernel
    // call kernel function with griddim and blockdim
    // pass device memory, rows, and cols
    ElementWiseMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // 7. Copy a result from device to host using cudaMemcpy
    // only need to copy back for C because thats the result
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // print to check the result
    for (int i = 0; i < rows * cols; i++) {
        std::cout << "C[" << i << "]" << h_C[i] << '\n';
    }

    // 8. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 9. Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}