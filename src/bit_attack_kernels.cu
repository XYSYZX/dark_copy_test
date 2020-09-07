#include "bit_attack.h"
#include "cuda.h"

__global__ void sign_attacker_kernel(float *x, float *grad, int n, float epsilon)
{
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    if(index >= n) return;
    if(*grad > .000001) *x += epsilon;
    else if(*grad < -0.000001) *x -= epsilon;
    else *x = *x;
}

extern "C" void sign_attacker_gpu(float *x_gpu, float *loc, int topk, float *grad_gpu, float epsilon)
{
    printf("attack sign gpu!\n");
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
    if(*grad > .000001) *x -= epsilon;
    else if(*grad < -0.000001) *x += epsilon;
    else *x = *x;
}

extern "C" void sign_delete_gpu(float *x_gpu, float *loc, int topk, float *grad_gpu, float epsilon)
{
    printf("delete sign gpu!\n");
    int i, idx;
    for(i = 0; i < topk; i++){
        idx = loc[i];
        sign_delete_kernel<<<1, 1>>>(&x_gpu[idx], &grad_gpu[idx], 1, epsilon);
    }
}

