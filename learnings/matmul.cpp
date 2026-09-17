#include <bits/stdc++.h>

using namespace std;

#define int long long
#define N 10

//version 1
//gpu/kernel launch also addds some overhead.
__global__ matmul_optimizing_i_loop(int* matrix, int* result){
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int j = 0;j<N; j++){
        for(int k = 0;k<N; k++){
            result[x][k] = matrix[x][k] * matrix[k][j];
        }
    }
}

//for the below, <<N,N>>, this is structure for blocks we will have.
__global__ matmul_both_outer_loops(int* matrix, int* result){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int i = id / N;
    int j = id % N;

    for(int k = 0;k<N; k++){
        // ops/
        //
    }
}

signed main(){
    //N == N 
    vector<vector<int>> matrix(N, vector<int>(N));
    for(int i =0;i<N; i++) for(int j = 0;j<N; j++) matrix[i][j] = i*N+j;

    //now simple matrix multiplication code
    for(int i = 0;i< N; i++){
        for(int j = 0;j<N; j++){

        
            for(int k = 0; k<N; k++){
                matrix[i][j] += matrix[i][k] * matrix[k][j];
            }
        }
    }
    
    //now optimize this code.
    //while optmizing this code, we always looks for loops, here we wont be parallelising `inner` most loop, as for multiple values of K we are accessing saem location, so race condition can be there, so we willbe focusing on outer most two loops.
    //Now the qus is which one to choose i or j?
    //Choosing i is a good choice, as we are accessing the memory in row major forms, so caching can help us in fetching multiple j's together.
    
    matmul_optimizing_i_loop(matrix, matrix);    
}
