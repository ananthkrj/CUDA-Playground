#include <iostream>
#include <cuda_runtime.h>

// create kernel
// lets make A, B, and C double data types
__global__ void NaiveMatMult(double* A, double* B, double* C, int N)
{
    // write a 2 dimensional calculation
    // 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // N is the size of the matrices
    // A, B, and C are matrices. They are of the size
    // N x N

    // REP is a benchmarking optimization trick
    // repeats the same multiplication to simulate a heavier workload
        // bounds check, make sure col and rows
        // are less than the size of matrices
        if (col < N && row < N) {
            // move the sum initiazliation inside
            double sum = 0.0;
            // now do matrix multiplication calculation
            // need to iterate over N x N
            for (int i = 0; i < N; i++) {
                // add up the dot products
                // figure out why there is a difference
                // in the dot product calculations here
                sum += A[row * N + i] * B[i * N + col];
            }
            // calculation for C
            C[row * N + col] = sum;
        }
}


// main function to execute kernel 
// transfer from host 
int main() {
    // initialize the variables being used
    // with corresponding values
    // N is just ... which means N x N
    int N = 10;
    int row = N;
    int col = N;
    int size = row * col * sizeof(double);

    // allocate host memory
    // allocate each matrix with the size
    // we just declared
    double* h_A = (double*)malloc(size);
    double* h_B = (double*)malloc(size);
    double* h_C = (double*)malloc(size);    

    // allocate device memory (cuda)
    // first declare the device variables
    // pass device variable by address
    double* d_A;
    cudaMalloc(&d_A, size);
    double* d_B;
    cudaMalloc(&d_B, size);
    double* d_C;
    cudaMalloc(&d_C, size);

    // copy from host to device (cudaMemcpy)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // configure block and grid dimensions (using blockdim and gridim)
    // This launch enough threads to cover the elements
    dim3 blockDim(16, 16);
    dim3 gridDim((col + 16) / 15, (row + 16) / 15);

    // launch the kernel (call kernel function)
    // matrix variables should be called as the device versions
    // as we are compiling on the device
    NaiveMatMult<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // copy back from device to host for the computing value
    // which is C
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // print result
    // always print on host, because right before this
    // we are copying from device to host
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // print on the host
            std::cout << h_C[i * N + j];
        }
        std::cout << '\n';
    }

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}