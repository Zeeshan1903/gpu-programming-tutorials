#include <cuda_runtime.h>
#include <stdio.h>

/*
    Memory levels on gpu
    Registers --> They are per thread level, so each thread will have some set
   of registers and they are extremely fast.
   SharedMemory --> BlockLevel, very
   fast, so all threads inside one block can share this memory
   L1 cache --> SM level, so all blocks which are executed by one sm will have
   access to l1 cache. L2 cache --> global level

    Note: When we do cudamalloc() --> global level...
*/

/*
    Shared Memory
        We can make memory shared by usnig __shared__ Keyword, which works at
   block level.

   So to avoid race conditions among threads from same block we use
   __syncthreads(), which says dont go beyond this point, if you have not
   completed all the fns.
*/

// we generlly follow SIMT, Single Instruction Multiple thread, meaning multiple
// trheads on diff core performing same inst.

// CudaGetLastError() to get last error for cuda.

/*
    What happens at gpu level, when we do vector addition
    On GPU's we have SM's and each sm's can execute blocks, each blocks have
   some threads, but at hardware level we have something called as warp, which
   is group of 32 threads, so say each block has 64 threads, so it means each
   block now have 2 warp and this warp will be executed in one go.

    so when we do
    __global__ vectorAdd(int* a, int* b,int* c, int n){
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx < n)c[idx] = a[idx] + b[idx];

        //now above 32 threads will run in parallely and will do/perform same
   inst ie additino, this is simt architecture.
        //So a block is a cuda programming abs whereas a warp is actual hardware
   level abstraction.
        //
    }


                SM
    +---------------------------+
    | Warp schedulers           |
    |                           |
    | CUDA cores / execution    |
    | units                     |
    |                           |
    | Register file             |
    |                           |
    | Shared memory / L1        |
    +---------------------------+
*/
int main() {}
