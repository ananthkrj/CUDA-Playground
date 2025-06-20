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

// stencil kernel
__global__ void StencilKernel(float* input, float* output, int N, int M) {
    // N in row, M is column
    // Shared memory + halo, but what are halo's, halo regions?

    // initialize col and rows using threadidx, blockidx and 
    // TILE_SIZE to represent the block dimensions

    // loading into shared memory with halo, do this for all the 
    // halo borders. 
    // Each shared memory loading for different halos needs
    // different edge case checking
    // In order by steps:
    // 1. The Main tile
    // 2. Horizontal halos
    // 3. Vertical halos

    // Then sync all these threads together

    // now perform the stencil operation
    // if in the valid region


}

// helper function
void launchStencilKernel() {

}

// main launch function
int main() {
    // declare N and M, 
    // N is row and M is column

    // allocate size

    // allocate host memory

    // call helper launch function
}