#include <iostream>
#include <cuda_runtime.h>

// create kernel
// lets make A, B, and C double data types
__global__ void NaiveMatMult(double* A, double* B, double* C, int N, int Rep)
{
    // write a 2 dimensional calculation
    // 
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    int cols = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize a sum variable, because in
    // matrix multiplication we are summing the dot
    // products
    double sum = 0.0;


    // N is the size of the matrices
    // A, B, and C are matrices. They are of the size
    // N x N

    // REP is a benchmarking optimization trick
    // repeats the same multiplication to simulate a heavier workload
    for (int r = 0; r < REP; r++) {
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
}


// main function to execute kernel 
// transfer from host 
int main() {
    // initialize matrix dimensions

    // allocate host memory

    // allocate device memory (cuda)

    // copy from host to device (cudamalloc)

    // configure block and grid dimensions (using blockdim and gridim)
    // launch enough threads to cover the elements

    // launch the kernel (call kernel function)

    // copy back from device to host for the computing value
    // which is C

    // print result

    // free device memory

    // free host memory
}