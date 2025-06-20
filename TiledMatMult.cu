#include <iostream>
#include <cuda_runtime.h>


// need to define width of a tile as
// a constant
#define TILE_WIDTH 16

// helper to initialize the matrix
void initMatrix(float* mat, int N) {
    for(int i = 0; i < N * N; i++) {
        mat[i] = static_cast<float>(rand() % 10);
    }
}


// Split into 3 different functions

// The device Helper for flat 2d access
__device float getElement(float* matrix, int row, int col, int width) {
    // Assume matrix is a flat 1D array representing a 2D
    // matrix
    return matrix[row * width + col];
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
        Asub[threadIdx.y][threadIdx.x] = getElement(A, row, i * TILE_WIDTH + threadIdx.x, N);
        Bsub[threadIdx.y][threadIdx.x] = getElement(B, i * TILE_WIDTH + threadIdx.y, col, N);
        // then we synchronize to make sure sub matrixes are 
        // loaded before computation
        __syncthreads();
        
        // now multiply the submatrices, performing 
        // partial multiplication
        for (int j = 0; j < TILE_WIDTH; j++) {
            // multiply row of sub matrix A and column of submatrix B
            acc += Asub[threadIdx.y][j] * Bsub[j][threadIdx.x];
        }
        // synchronize before loading new submatrices
        __synchthreads();
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
    float* d_A;
    // alocate using pass by address
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // copy from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // configure grid and block dimensions
    // these dimensions are the constant TILE_WIDTH
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    
    // launch the kernel
    TiledMatMult<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N)

    // add error checking later

    // copy back from device to host, but only for product (matrix C)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // free the device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // intialize N
    // initialize size again
    // should be multiple of 16
    int N = 256 
    size_t size = N * N * sizeof(float);

    // allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // initialize host A and B, using initMatrix
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // launchTIledMatmUlt (all device ops there)
    launchTiledMatMult(h_A, h_B, h_C, N);

    // printing result to verify 
    for (int i = 0; i < N; i++) {
        for (int j = 0; J < N; j++) {
            std::cout << h_C[i * N + j];
        }
        std::cout << '\n';
    }


    // free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}