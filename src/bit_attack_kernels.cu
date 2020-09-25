#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "bit_attack.h"
}
__global__ void sign_attacker_kernel(float *x, float *grad, int n, float epsilon)
{
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    if(index >= n) return;
    if(*grad > .000001) *x -= epsilon;
    else if(*grad < -0.000001) *x += epsilon;
    else *x = *x;
}

void sign_attacker_gpu(float *x_gpu, int *loc, int topk, float *grad_gpu, float epsilon)
{
    //printf("attack sign gpu!\n");
    int i, idx;
    for(i = 0; i < topk; i++){
        idx = loc[i];
        sign_attacker_kernel<<<1, 1>>>(&x_gpu[idx], &grad_gpu[idx], 1, epsilon);
    }
}
__global__ void sign_delete_kernel(float *x, float *grad, int n, float epsilon)
{
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    if(index >= n) return;
    if(*grad > .000001) *x += epsilon;
    else if(*grad < -0.000001) *x -= epsilon;
    else *x = *x;
}

void sign_delete_gpu(float *x_gpu, int *loc, int topk, float *grad_gpu, float epsilon)
{
    //printf("delete sign gpu!\n");
    int i, idx;
    for(i = 0; i < topk; i++){
        idx = loc[i];
        sign_delete_kernel<<<1, 1>>>(&x_gpu[idx], &grad_gpu[idx], 1, epsilon);
    }
}

void bit_flip_attacker_gpu(attack_args a)
{
    int idx = a.mloss_loc[a.layer_idx][a.k_idx];
    float *x = a.x_gpu[a.layer_idx];
    int bit_idx = a.bit_idx;
    inject_noise_float_onebit_gpu(x, idx, bit_idx);
}


