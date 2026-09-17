#include <cuda_runtime.h>
#include <stdio.h>

/*
    When we use
    __global__: code runs on gpu called from cpu
    __device__: code runs on gpu called from gpu
    __host__  : code run on cpu and called from cpu

*/
__device__ void callHelloWorldFromGPU() { printf("Hello World\n"); }
__global__ void helloWorld() { callHelloWorldFromGPU(); }

int main() {

  // here calling the thread with B blocks, and each block having T threads
  // Now each thread will have T threads and they will be scheduled on SM's
  // (Streaming Multiprocessor), which are execution unit on gpu's Now each
  // thread is not scheduled on sm's instead they are grouped together which is
  // called as warp and that warp is scheduled on sm's Each warp consists of 32
  // threads We are using 68 sm's Grid ---> Block ---> threads
  helloWorld << 1, 32 >>();

  // always remember kernel call is asynchrnous.
  cudaDeviceSynchronize();

  return 0;
}
