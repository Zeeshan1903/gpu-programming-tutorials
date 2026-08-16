#include <cuda_runtime.h>
#include <iostream>

__global__ void barrierKernel(int *counter, int *result) {

  int tid = threadIdx.x;

  atomicAdd(counter, 1);

  __syncthreads();

  // Thread 0 waits until all threads have incremented counter
  if (tid == 0) {
    while (*counter < blockDim.x) {
    }

    *result = 1;
  }

  // this is to check whether all threads are synced or not.
  __syncthreads();

  printf("Thread %d passed the barrier\n", tid);
}

int main() {

  int *d_counter;
  int *d_result;

  int counter = 0;
  int result = 0;

  cudaMalloc(&d_counter, sizeof(int));
  cudaMalloc(&d_result, sizeof(int));

  cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

  int threads = 8;

  barrierKernel<<<1, threads>>>(d_counter, d_result);

  cudaDeviceSynchronize();

  cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "Barrier completed: " << (result ? "Yes" : "No") << std::endl;

  cudaFree(d_counter);
  cudaFree(d_result);

  return 0;
}
