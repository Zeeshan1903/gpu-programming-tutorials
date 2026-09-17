#include <cuda_runtime.h>
#include <stdio.h>

// having a fn that runs on gpu and performs vector addition

__global__ vectorAdd(int *a, int *b, int *c, int n) {

  // a[i] + b[i] = c[i];
  // find index i:
  //   i = blockIdx.x * blockDim.x + threadIdx.x
  //   why because each block wll have blcok id and thread id
  //   block0
  //                   thread0                 0 * 2 + 0
  //                   thread1                 0 * 2 + 1
  //                   thread2                 0 * 2 + 2
  //   block1
  //                   thread0                 1 * 2 + 0
  //                   thread1                 1 * 2 + 1
  //                   thread2                     same
  //  blockDim.x == no of blocks which is 2 in above case.
  //  Now i = dim * current_block_id + offset_inside_that_block(thread_id)

  // we can have many threads, but the number of ele might not be equal to that
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int n = 10;
  int *h_a = malloc(sizeof(int) * n);
  int *h_b = malloc(sizeof(int) * n);
  int *h_c = malloc(sizeof(int) * n);

  // intialize
  for (int i = 0; i < n; i++)
    h_a[i] = i * 2 + 1;
  for (int i = 0; i < n; i++)
    h_b[i] = i * 3 + 1;

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, n * sizeof(int));
  cudaMalloc(&d_b, n * sizeof(int));
  cudaMalloc(&d_c, n * sizeof(int));

  // now transfer data from cpu to gpu side..
  // cpu --> gpu
  // cudaMemcpy(destination, source, size, direction)
  cudaMemcpy(d_a, h_a, sizeof(int) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int) * n, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (n + threads + 1) / threads;

  vectorAdd << blocks, threads >> (d_a, d_b, d_c, n);

  // to synchronize or get the output from kernel, basically we are waiting for
  // kernel code to fnish.
  cudaDeviceSynchronize();

  // nw get our answer from gpu side
  // gpu --> side
  cudaMemcpy(h_c, d_c, sizeof(int) * n, cudaMemcpyDeviceToHost);

  cudaFree(d_a); // similar for others.
  free(h_a);
}
