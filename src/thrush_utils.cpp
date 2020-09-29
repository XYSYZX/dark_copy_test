#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "darknet.h"

extern "C"{
#include "cuda.h"
}


void get_topk_gpu(float *x_gpu, int length, int *y_idx, int topk)
{
    abs_gpu(x_gpu, x_gpu, length);

    thrush::device_vector<int> x_dev_idx[length];
    //int *x_gpu_idx = thrust::raw_pointer_cast(x_dev_idx);

    thrush::sequence(x_dev_idx.begin(), x_dev_idx.end());
    thrust::device_ptr<float> x_dev(x_gpu);
    thrust::sort_by_key(thrush::device, x_dev, x_dev + length, x_dev_idx, thrust::greater<float>());
    //thrust::copy(x_dev, x_dev + topk, y);
    cuda_pull_array(x_dev_idx, y_idx, topk);
}

