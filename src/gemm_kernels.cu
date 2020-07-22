#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"


extern "C" {
#include "gemm.h"
#include "cuda.h"
}

__global__ void gemm_nn_cuda(const float* A, const float* B, float* C, 
			     const int M, const int N, const int K, float ALPHA, float BETA)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0;
	if ( row < M && col < N ) {
		C[row * N + col] *= BETA;
		for (int i = 0; i < K; i++) {
			sum += ALPHA * A[row * K + i] * B[i * N + col];
		}
		C[row * N + col] += sum;
	} 
}

__global__ void gemm_nt_cuda(const float* A, const float* B, float* C, 
			     const int M, const int N, const int K, float ALPHA, float BETA)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0;
	if ( row < M && col < N ) {
		C[row * N + col] *= BETA;
		for (int i = 0; i < K; i++) {
			sum += ALPHA * A[row * K + i] * B[col * K + i];
		}
		C[row * N + col] += sum;
	} 
}

__global__ void gemm_tn_cuda(const float* A, const float* B, float* C, 
			     const int M, const int N, const int K, float ALPHA, float BETA)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0;
	if ( row < M && col < N ) {
		C[row * N + col] *= BETA;
		for (int i = 0; i < K; i++) {
			sum += ALPHA * A[i * M + row] * B[i * N + col];
		}
		C[row * N + col] += sum;
	} 
}

__global__ void gemm_tt_cuda(const float* A, const float* B, float* C, 
			     const int M, const int N, const int K, float ALPHA, float BETA)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0;
	if ( row < M && col < N ) {
		C[row * N + col] *= BETA;
		for (int i = 0; i < K; i++) {
			sum += ALPHA * A[i * M + row] * B[col * K + i];
		}
		C[row * N + col] += sum;
	} 
}

extern "C" void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    //int grid_rows = (M + BLOCK - 1) / BLOCK;
    //int grid_cols = (N + BLOCK - 1) / BLOCK;
    //dim3 dimGrid(grid_cols, grid_rows);
    //dim3 dimBlock(BLOCK, BLOCK);
    if(!TA && !TB)
    	gemm_nn_cuda<<<dimGrid, dimBlock>>>(A_gpu, B_gpu, C_gpu, M, N, K, ALPHA, BETA);
    else if(TA && !TB)
    	gemm_tn_cuda<<<dimGrid, dimBlock>>>(A_gpu, B_gpu, C_gpu, M, N, K, ALPHA, BETA);
    else if(!TA && TB)
    	gemm_nt_cuda<<<dimGrid, dimBlock>>>(A_gpu, B_gpu, C_gpu, M, N, K, ALPHA, BETA);
    else
    	gemm_tt_cuda<<<dimGrid, dimBlock>>>(A_gpu, B_gpu, C_gpu, M, N, K, ALPHA, BETA);
}

