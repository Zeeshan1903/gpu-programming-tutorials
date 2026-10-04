#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define N    64          // matrix dimension (N x N)
#define TILE 16          // tensor core tile size (16x16x16)

// Each block = 1 warp (32 threads) = computes one 16x16 tile of C
__global__ void wmma_matmul(const half *A, const half *B, float *C)
{
    int tileRow = blockIdx.y;   // which row-tile of C (0..3)
    int tileCol = blockIdx.x;   // which col-tile of C (0..3)

    // Fragments live in registers, spread across the 32 threads of the warp
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // C(tile) = sum over k of A(tileRow,k) * B(k,tileCol)
    for (int k = 0; k < N / TILE; k++) {
        const half *aPtr = A + (tileRow * TILE) * N + (k * TILE);
        const half *bPtr = B + (k * TILE) * N + (tileCol * TILE);

        wmma::load_matrix_sync(a_frag, aPtr, N);   // N = leading dimension
        wmma::load_matrix_sync(b_frag, bPtr, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // tensor core op
    }

    float *cPtr = C + (tileRow * TILE) * N + (tileCol * TILE);
    wmma::store_matrix_sync(cPtr, c_frag, N, wmma::mem_row_major);
}

// CPU reference
void cpu_matmul(const float *A, const float *B, float *C)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main()
{
    size_t elems = N * N;

    float *hA_f = new float[elems], *hB_f = new float[elems];
    float *hC_gpu = new float[elems], *hC_cpu = new float[elems];
    half  *hA = new half[elems],     *hB = new half[elems];

    // Small values: exactly representable in FP16, so results should match
    for (int i = 0; i < N * N; i++) {
        hA_f[i] = (float)(i % 5) * 0.5f;
        hB_f[i] = (float)(i % 3) * 0.25f;
        hA[i] = __float2half(hA_f[i]);
        hB[i] = __float2half(hB_f[i]);
    }

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, elems * sizeof(half));
    cudaMalloc(&dB, elems * sizeof(half));
    cudaMalloc(&dC, elems * sizeof(float));
    cudaMemcpy(dA, hA, elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, elems * sizeof(half), cudaMemcpyHostToDevice);

    dim3 grid(N / TILE, N / TILE);   // 4 x 4 = 16 tiles
    dim3 block(32, 1);               // one warp per tile
    wmma_matmul<<<grid, block>>>(dA, dB, dC);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(hC_gpu, dC, elems * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    cpu_matmul(hA_f, hB_f, hC_cpu);
    float maxErr = 0.0f;
    for (int i = 0; i < N * N; i++)
        maxErr = fmaxf(maxErr, fabsf(hC_gpu[i] - hC_cpu[i]));

    printf("Max abs error (GPU tensor core vs CPU): %f\n", maxErr);
    printf("C[0][0] GPU=%f CPU=%f\n", hC_gpu[0], hC_cpu[0]);
    printf("%s\n", maxErr < 1e-2f ? "TEST PASSED" : "TEST FAILED");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA_f; delete[] hB_f; delete[] hC_gpu; delete[] hC_cpu;
    delete[] hA; delete[] hB;
    return 0;
}
