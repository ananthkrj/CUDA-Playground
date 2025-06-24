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
#define TILE_SIZE 16;
// Shared memory arrays need to be on compile time
#define HALO_SIZE 1;
// Shared size is essentially just combining two tiled (times 2)
#define SHARED_SIZE (TILE_SIZE + 2 * HALO_SIZE)

// stencil kernel
__global__ void StencilKernel(float* input, float* output, int N, int M) {
    // N in row, M is column
    // compute the shared memory
    __shared__ float shared_Data[SHARED_SIZE][SHARED_SIZE];

    // initialize col and rows using threadidx, blockidx and 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory coordinates (where to store the shared memory)
    int ty = threadIdx.y * HALO_SIZE;
    int tx = threadIdx.x * HALO_SIZE;

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
            shared_data[ty][tx - HALO_SIZE] = input[row * M + col];
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
            shared_data[ty][col + HALO_SIZE] = input[row * M + col];
        // out of bounds halo
        } else {
            shared_data[ty][col + HALO_SIZE] = input[row * N + col];
        }
    }

    // load vertical tiles (the ones that are top and bottom
    // around main tile)
    // do top first
    if (threadIdx.y == 0)

    // Then sync all these threads together

    // now perform the stencil operation
    // if in the valid region


}

// helper function
void launchStencilKernel() {
    // declare size

    // initilize device variables

    // Use CudaMemcpy to copy from host to device memory

    // Initialiize blockdim and griddim

    // call stencilkernel using blokcdim gridim, and the kernel parameters

    // add error checking for kernel call

    // copy from device to host (overall value)

    // free device memory
}

// main launch function
int main() {
    // declare N and M, 
    // N is row and M is column
    // 

    // allocate size

    // allocate host memory

    // call helper launch function
}