#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "cuda.h"
#include "bound_adjust.h"
#include "darknet.h"
}

__global__ void find_max_kernel(float *x, float *gx, int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    float *tmp_x = x + (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x;
    if(i > n) return;

    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        if((tid % (2 * stride)) == 0) {
            if(tmp_x[tid] < tmp_x[tid + stride])
                tmp_x[tid] = tmp_x[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0) gx[blockIdx.x + blockIdx.y * gridDim.x] = tmp_x[0];
}

void find_max_gpu(float *x, int n, float *max)
{
    int block_n = cuda_grid_xyz(n);
    float *y = (float *)calloc(block_n, sizeof(float));
    float *gy = cuda_make_array(y, block_n);
    find_max_kernel<<<cuda_gridsize(n), BLOCK>>>(x, gy, n);
    cuda_pull_array(gy, y, block_n);
    for(int i = 0; i < block_n; i++){
        if(*max < y[i])  *max = y[i];
    }
    cuda_free(gy);
    free(y);
}

__global__ void find_min_kernel(float *x, float *gx, int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    float *tmp_x = x + (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x;
    if(i > n) return;

    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        if((tid % (2 * stride)) == 0) {
            if(tmp_x[tid] > tmp_x[tid + stride])
                tmp_x[tid] = tmp_x[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0) gx[blockIdx.x + blockIdx.y * gridDim.x] = tmp_x[0];
}

void find_min_gpu(float *x, int n, float *min)
{   
    int block_n = cuda_grid_xyz(n);
    float *y = (float *)calloc(block_n, sizeof(float));
    float *gy = cuda_make_array(y, block_n);
    find_min_kernel<<<cuda_gridsize(n), BLOCK>>>(x, gy, n);
    cuda_pull_array(gy, y, block_n);
    for(int i = 0; i < block_n; i++){
        if(*min > y[i])  *min = y[i];
    }
    cuda_free(gy);
    free(y);
}

void get_output_bound_gpu(layer l)
{
    int num = l.outputs*l.batch;
    float min = 3.402823466e+38F;
    float max = 1.175494351e-38F;
    float *output_gpu_tmp = cuda_make_array_dev(l.output_gpu, num);
    find_max_gpu(output_gpu_tmp, num, &max);
    cuda_push_array_dev(output_gpu_tmp, l.output_gpu, num);
    find_min_gpu(output_gpu_tmp, num, &min);
    if(max > *l.max_output) *l.max_output = max;
    if(min < *l.min_output) *l.min_output = min;
    printf("layer index: %d, max: %.5f, min: %.5f\n", l.index, *l.max_output, *l.min_output);
    cuda_free(output_gpu_tmp);
}

void get_weight_bound_gpu(layer l)
{
    int num = l.c/l.groups*l.n*l.size*l.size;
    float min = 3.402823466e+38F;
    float max = 1.175494351e-38F;
    float *weights_gpu_tmp = cuda_make_array_dev(l.weights_gpu, num);
    find_max_gpu(weights_gpu_tmp, num, &max);
    cuda_push_array_dev(weights_gpu_tmp, l.weights_gpu, num);
    find_min_gpu(weights_gpu_tmp, num, &min);
    if(max > *l.max_weight) *l.max_weight = max;
    if(min < *l.min_weight) *l.min_weight = min;
    printf("max: %.5f, min: %.5f\n", *l.max_weight, *l.min_weight);
    cuda_free(weights_gpu_tmp);
}

__global__ void constrain_output_kernel(float *x, float max, float min, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i > n) return;
    if(x[i] > max) x[i] = max;
    if(x[i] < min) x[i] = min;
}

void constrain_output_gpu(float *x, float max, float min, int n)
{
    constrain_output_kernel<<<cuda_gridsize(n), BLOCK>>>(x, max, min, n);
}
__global__ void constrain_weight_kernel(float *x, float max, float min, int n, int layer)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i > n) return;
    int j = 0;
    float temp;
    if(x[i] > max) {
	temp = x[i];
	x[i] = max - 0.000001;
	j = 1;
    }
    if(x[i] < min) {
	temp = x[i];
	x[i] = min + 0.000001;
	j = 1;
    }
    if(j == 1) {
	 //fprintf(stderr, "value of weight is changed!\n");
	 printf("layer: %d, index: %d, value: %f\n", layer, i, temp);
    }
}

void constrain_weight_gpu(float *x, float max, float min, int n, int layer)
{
    constrain_weight_kernel<<<cuda_gridsize(n), BLOCK>>>(x, max, min, n, layer);
}


void check_weights_gpu(network *net)
{
    for(int i = 0; i < net->n; i++){
        if(net->layers[i].type == CONVOLUTIONAL){
            layer l = net->layers[i];
            int n = l.c / l.groups * l.n * l.size * l.size;
            float max = *l.max_weight;
            float min = *l.min_weight;
            constrain_weight_gpu(l.weights_gpu, max, min, n, i);
        }
    }
}

void check_outputs_gpu(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == MAXPOOL || l.type == ROUTE || l.type == SHORTCUT || l.type == UPSAMPLE){
        int n = l.outputs*l.batch;
        float max = *l.max_output;
        float min = *l.min_output;
        constrain_output_gpu(l.output_gpu, max, min, n);
        //printf("finish check output gpu\n");
    }
}

