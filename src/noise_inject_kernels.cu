#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
extern "C" {
#include "cuda.h"
#include "noise_inject.h"
//#include "darknet.h"
}

int *generate_random_int(int *rand_array, int n)
{
    int *rand_array_gpu = cuda_make_int_array(rand_array, n);
    cuda_random_int(rand_array_gpu, n);
    return rand_array_gpu;
}    

__global__ void single_bit_flip_float_kernel(float *w_gpu, unsigned int length, int noise_freq,unsigned int *bit_array_gpu)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    //int index = threadIdx.x + blockDim.x * threadIdx.y;
    if(index >= length) return;
    //printf("index: %d\n", index);
    if(index % noise_freq == 0){
        unsigned int idx = bit_array_gpu[index] & 31;  //a%32 = a&(32-1)
        //printf("idx: %d\n", idx);
        unsigned int b = 0x1;
        unsigned int new_b;
        unsigned int *p;
        p = (unsigned int *)(&w_gpu[index]);
        new_b = b << idx;
        //printf("chang b to: %d\n", new_b);
        (*p) = (*p)^(new_b);
        //printf("chang w to: %f\n", w_gpu[index]);
        __syncthreads();
    }
}

void single_bit_flip_float_gpu(float *w_gpu, unsigned int n, int noise_freq)
{
    int *bit_array = (int *)calloc(n, sizeof(int));
    int *bit_array_gpu = generate_random_int(bit_array, n);
    single_bit_flip_float_kernel<<<cuda_gridsize(n), BLOCK>>>(w_gpu, n, noise_freq, (unsigned int*)bit_array_gpu);
    //single_bit_flip_float_kernel<<<1, 16>>>(w_gpu, n, noise_freq, (unsigned int*)bit_array_gpu);
    check_error(cudaPeekAtLastError());
    cuda_free_int(bit_array_gpu);
    free(bit_array);
}

void inject_noise_float_gpu(float *w_gpu, unsigned int length)
{
    int noise_freq = rand()%NOISE_FREQ + NOISE_FREQ;
    //printf("noise_freq: %d\n", noise_freq);
    single_bit_flip_float_gpu(w_gpu, length, noise_freq);
}

__global__ void single_bit_flip_float_limit_kernel(float *w_gpu, unsigned int length, int *limit_gpu, int limits, int noise_freq, unsigned int *bit_array_gpu)
{
    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    //int index = threadIdx.x + blockDim.x * threadIdx.y;
    if(index >= length) return;
    //printf("index: %d\n", index);
    if(index%noise_freq==0){
        unsigned int idx = bit_array_gpu[index] & 31;
        //printf("idx: %d\n", idx);
        while(1){
            int flag = 0;
            for(int i=0; i < limits; i++){
                if(idx==limit_gpu[i]){
                    flag = 1;
                    idx = (idx + bit_array_gpu[index] & 31) & 31;
                    //printf("change idx: %d\n", idx);
                    break;
                }
            }
            if(flag == 0){
                unsigned int b = 0x1;
                unsigned int new_b;
                unsigned int *p;
                p = (unsigned int *)(&w_gpu[index]);
                new_b = b << idx;
                //printf("chang b to: %d\n", new_b);
                (*p) = (*p)^(new_b);
                //printf("chang w to: %f\n", w_gpu[index]);
                break;
            }
        }
    }
}

void single_bit_flip_float_limit_gpu(float *w_gpu, unsigned int n, int *limit_gpu, int limits, int noise_freq)
{
    int *bit_array = (int *)calloc(n, sizeof(int));
    int *bit_array_gpu = generate_random_int(bit_array, n);
    single_bit_flip_float_limit_kernel<<<cuda_gridsize(n), BLOCK>>>(w_gpu, n, limit_gpu, limits, noise_freq, (unsigned int*)bit_array_gpu);
    //single_bit_flip_float_limit_kernel<<<1, 16>>>(w_gpu, n, limit_gpu, limits, noise_freq, (unsigned int*)bit_array_gpu);
    check_error(cudaPeekAtLastError());
    free(bit_array);
    cuda_free_int(bit_array_gpu);
}

void inject_noise_float_limit_gpu(float *w_gpu, unsigned int length, int *limit_gpu, int limits)
{
    int noise_freq = rand()%NOISE_FREQ + NOISE_FREQ;
    //printf("noise_freq: %d\n", noise_freq);
    single_bit_flip_float_limit_gpu(w_gpu, length, limit_gpu, limits, noise_freq);
}

__global__ void single_bit_flip_float_onebit_kernel(float *w_gpu, int length, int weight_idx, int bit_idx)
{
	int index = blockIdx.x *blockDim.x + threadIdx.x;
	if(index >= length) return;
    unsigned int b = 0x1;
    unsigned int new_b;
    unsigned int *p;
    p = (unsigned int *)(&w_gpu[weight_idx]);
    new_b = b << bit_idx;
    //printf("chang b to: %d\n", new_b);
    (*p) = (*p)^(new_b);
}

void inject_noise_float_onebit_gpu(float *w_gpu, int weight_idx, int bit_idx)
{
	//float *w_gpu = &w_gpu[weight_idx];
	single_bit_flip_float_onebit_kernel<<<1, 1>>>(w_gpu, 1, weight_idx, bit_idx);
}
	


void test_inject_noise_gpu()
{
    int length = 16;
    float *w = (float *)calloc(length, sizeof(float));
    float *y = (float *)calloc(length, sizeof(float));
    for(int i=0; i<length; i++){
        w[i] = 1.0;
        y[i] = 1.0;
    }
    int limit[4] = {31, 30, 29, 28};
    int limits = 4;
     
    float *w_gpu = cuda_make_array(w, length);
    float *y_gpu = cuda_make_array(y, length);
    int *limit_gpu = cuda_make_int_array(limit, limits);
    inject_noise_float_gpu(w_gpu, length);
    cuda_pull_array(w_gpu, w, length);
    for(int i=0; i<16; i++){
        printf("%.16f ", w[i]);
    }
    printf("\n");
    inject_noise_float_limit_gpu(y_gpu, length, limit_gpu, limits);
    cuda_pull_array(y_gpu, y, length);
    for(int i=0; i<16; i++){
        printf("%.16f ", y[i]);
    }
    cuda_free(w_gpu);
    cuda_free(y_gpu);
    cuda_free_int(limit_gpu);
    free(w);
    free(y);
}
