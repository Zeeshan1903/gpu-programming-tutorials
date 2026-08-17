#include <stdio.h>
#include <cuda_runtime.h>

__device__ int blockCounter = 0;

__global__ void gridBarrier(int totalBlocks) {

    // One thread from each block arrives at the barrier
    if (threadIdx.x == 0) {
        atomicAdd(&blockCounter, 1);
    }

    // Make sure all threads in this block have reached this point
    __syncthreads();

    // Thread 0 of each block waits until every block has arrived
    if (threadIdx.x == 0) {
        while (atomicAdd(&blockCounter, 0) < totalBlocks) {
            // Wait
        }
    }

    // Release all threads in the block
    __syncthreads();

    printf("Block %d | Thread %d | Counter = %d\n",
           blockIdx.x,
           threadIdx.x,
           blockCounter);
}

int main() {

    int blocks = 2;
    int threadsPerBlock = 10;

    gridBarrier<<<blocks, threadsPerBlock>>>(blocks);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
