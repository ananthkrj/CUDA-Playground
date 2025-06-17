#include <iostream>
#include <cuda_runtime.h>

// need to parameterize 3 values
__global__ ElementWiseMultiply(float* A, float* B, float* C, int rows, int cols) 
{
    // Main calculation
    // do calculation for row
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // do calculation for column
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // out of bounds check (prevention), to prevent threads
    if (row < rows && col < cols) {
        // convert from 2d to 1d 
        int idx = row * cols + col;
        // map idx to C and A, B
        C[idx] = A[idx] * B[idx];
    }


}
int main(void) 
{
    // 1. Define matrix dimensions

    // 2. Allocate and intialize host memory for A, B, and C

    // 3. Allocate device memory

    // 4. We copy inputs from host to device using cudaMemcpy

    // 5. Configure block and grid dimensions using blockDim
    // and gridDim

    // 6. Launch the kernel

    // 7. Copy a result from device to host using cudaMemcpy

    // 8. Free device memory

    // 9. Free host memory (optiinal)
}