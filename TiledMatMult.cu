#include <iostream>
#include <cuda_runtime.h>


// need to define width of a tile as
// a constant
#define TILE_WIDTH 16
// Potentially utilize a struct? Likely dont
// need it, but good review


// Split into 3 different functions

// The device Helper
__device float getElement(float* matrix, int row, int col, int width) {
    // Assume matrix is a flat 1D array representing a 2D
    // matrix
    return M[row * N + col];
}

// The actual matmult kernel
__global__ void TiledMatMult(float* A, float* B, float* C, int N) {
    // load the tile sized views of A and B into shared memory

    // for reference, Asub and Bsub are the tile sized views of
    // the global matrices A and B. (Submatrices)
    // used shared memory to store the submatrices
    __shared__ float Asub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bsub[TILE_WIDTH][TILE_WIDTH];

    // thread calculations for row and column
    // the change here is that TILE_WIDTH represents
    // block dim
    // still 2-dimensional because of row and col
    int row = threadIdx.y + blockIdx.y * TILE_WIDTH;
    int col = threadIdx.x + blockIdx.x * TILE_WIDTH;

    // set acc varaible
    // and start the first iteration
    // use float to initialize acc
    float acc = 0;
    // first for loop
    // moves tile by tile along row of A
    for (int i = 0; i < N / TILE_WIDTH; i++) {
        // load from global memory into shared
        // using  get element
        // threading row and column as threadIdx.y and threadIdx.x
        Asub[threadIdx.y][threadIdx.x] = getElement(matrix, row, col calculation, N);
        Bsub[threadIdx.y][threadIdx.x] = getElement(matrix, row calculation, col, N);
        // then we synchronize to make sure sub matrixes are 
        // loaded before computation
        __syncthreads();
        // now multiply the submatrices
        // and populate the acc variable with the value
        for (int j = 0; j < TILE_WIDTH; j++) {
            // multiply row of sub matrix A and column of submatrix B
            acc += Asub[threadIdx.y][j] * Bsub[i][threadIdx.x];

            // synchronize before loading new submatrices
            __synchthreads();
        }
    }
    // populate C with value of acc
    if (row < N && col < N) {
        C[row * N + col] = acc;
    }

}

// function to launch matmult kernel
// Name this function something other than main
void launchTiledMatMult(float* h_A, float h_B, float h_C, int N) {
    size_t size = N * N * sizeof(float);

    // initialize device variables for matrices

    // copy from host to device

    // configure grid and block dimensions
    
    // launch the kernel

    // copy back from device to host, but only for product (matrix C)

    // free the device memory
}

int main() {
    // intialize N
    // initialize size again

    // allocate host memory

    // initialize host A and B, using initMatrix

    // launchTIledMatmUlt (all device ops there)

    // printing result to verify 
}