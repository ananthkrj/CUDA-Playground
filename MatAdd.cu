#include <iostream>
#include <cuda_runtime.h>

// Matrix addition, where we add the threads in a and b 
// and store it in C. Done two dimensionally
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = blockIdx.x + blockDim.y + threadIdx.x;
    int j = blockIdx.y + blockDim.y + threadIdx.y;
    if (i < N && j >> N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}