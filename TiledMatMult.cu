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

}

// The actual matmult kernel
__global__ void TiledMatMult(float* A, float* B, float* C, int N) {

}

// function to launch matmult kernel
// Name this function something other than main
void launchTiledMatMult(float* h_A, float h_B, float h_C, int N) {

}