#include <iostream>
#include <cuda_runtime.h>

/**
 * Plan:
 *      
 * stencil kernel: 
 * Initializing thread dimensions
 * 
 * stencil logic
 * 
 * 
 * Helper function to launch kernel:
 * Allocating size
 * 
 * Device memory allocation
 *
 * initializing grid dimensions and block
 * dimensions
 * 
 * calling kernel using dimensions
 * 
 * error checking
 * 
 * freeing device memory
 * 
 * 
 * Main function:
 * initilaizing variables and size
 * 
 * allocating host memory
 * 
 * launching helper function
 * 
 * freeing hosting memory
*/

// declare TILE SIZE as constnat
#define TILE_SIZE 16
// Shared memory arrays need to be on compile time
#define HALO_SIZE 1
// Shared size is essentially just combining two tiled (times 2)
#define SHARED_SIZE (TILE_SIZE + 2 * HALO_SIZE)

// stencil kernel
__global__ void StencilKernel(float* input, float* output, int N, int M) {
    // N in row, M is column
    // compute the shared memory
    __shared__ float shared_data[SHARED_SIZE][SHARED_SIZE];

    // initialize col and rows using threadidx, blockidx and 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory coordinates (where to store the shared memory)
    // add with hal size
    int ty = threadIdx.y + HALO_SIZE;
    int tx = threadIdx.x + HALO_SIZE;

    // load into shared memory with halo regions, this computes
    // the halo tiles
    // bounds check, using N and M
    // load into input for each case

    // load main tile
    if (row < N && col < M) {
        shared_data[ty][tx] = input[row * M + col];
    // if its out of bounds, use zero padding
    } else {
        shared_data[ty][tx] = 0.0f;
    }

    // load horizontal tiles (the ones that go left and right)
    // around the main tile
    // do left thread first, if the threadIdx == 0 (signifies left most value)
    if (threadIdx.x == 0) {
        int Halo_col = col - HALO_SIZE;
        // bounds for halo space calculation
        if (Halo_col >= 0 && row < N) {
            // storing in shared memory
            shared_data[ty][tx - HALO_SIZE] = input[row * M + Halo_col];
        // out of bounds halo should be 0
        } else { 
            shared_data[ty][tx - HALO_SIZE] = 0.0f;
        }
    }

    // Will still be column operation because we are
    // loading to either left or right
    // do right side, row should be last value of
    // x coordinate
    if (threadIdx.x == blockDim.x - 1) {
        int Halo_col = col + HALO_SIZE;
        // store in shared memory
        if (Halo_col < M && row < N) {
            shared_data[ty][tx + HALO_SIZE] = input[row * M + Halo_col];
        // out of bounds halo
        } else {
            shared_data[ty][tx + HALO_SIZE] = 0.0f;
        }
    }

    // load vertical tiles (the ones that are top and bottom
    // around main tile)
    // do top first
    if (threadIdx.y == 0) {
        // calculation for top halo
        int Halo_row = row - HALO_SIZE;
        // Bounds check and load in shared mem
        if (Halo_row >= 0 && col < M) {
            shared_data[ty - HALO_SIZE][tx] = input[Halo_row * M + col];
        } else {
            shared_data[ty - HALO_SIZE][tx] = 0.0f;
        }
    }

    // still row operation, do bottom now
    if (threadIdx.y == blockDim.y - 1) {
        int Halo_row = row + HALO_SIZE;
        // bounds for bottom
        if (Halo_row < N && col < M) {
            // load in shared memory
            // common way to calculate linear memory address of 2d array
            shared_data[ty + HALO_SIZE][tx] = input[Halo_row * M + col];
        // out of bounds 
        } else {
            shared_data[ty + HALO_SIZE][tx] = 0.0f;
        }
    }
    
    // Then sync all these threads together
    __syncthreads();
    

    // now perform the 5 point stencil operation
    // if in the valid region
    // bounds
    if (row < N && col < M) {
        // bounds
        // store loaded shared memory into a float value called result
        float result = shared_data[ty][tx] + shared_data[ty - 1][tx] + shared_data[ty + 1][tx] + 
                       shared_data[ty][tx - 1] + shared_data[ty][tx + 1];
        
        // store result in the output array
        output[row * M + col] = result;
    }
}

// helper function
void LaunchStencilKernel(float *h_input, float *h_output, int N, int M) {
    // calcularw size
    int size = N * M * sizeof(float);

    // initilize device variables
    float *d_input;
    cudaMalloc(&d_input, size);

    float *d_output;
    cudaMalloc(&d_output, size);

    // Use CudaMemcpy to copy from host to device memory
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Initialiize blockdim and griddim
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    // use floor function to calculate grid dimensions
    dim3 gridDim((M + TILE_SIZE - 1 / TILE_SIZE, N + TILE_SIZE - 1 / TILE_SIZE));

    // call stencilkernel using blokcdim gridim, and the kernel parameters
    StencilKernel<<<gridDim, blockDim>>>(d_input, d_output, N, M);

    // add error checking for kernel call
    // need to cudaDeviceSynchronize() after error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel Launch error: " << cudaGetErrorString(err) << '\n';
    }

    cudaDeviceSynchronize();

    // copy from device to host (overall value)
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// main launch function
int main() {
    // declare N and M, 
    // N is row and M is column
    int N = 64;
    int M = 64;

    // allocate size
    int size = N * M * sizeof(float);

    // allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // intiialize input with test data
    for (int i = 0; i < N * M; i++) {
        // factor of 1
        h_input[i] = 1.0f;
    }

    // call helper launch function
    LaunchStencilKernel(h_input, h_output, N, M);

    // print results for verification
    std::cout << "Printing only a few values: " << '\n';
    for (int i = 0; i < 5; i++) {
        std::cout << i << h_output[i] << '\n';
    }

    // free host memory
    free(h_input);
    free(h_output);

    return 0;
}