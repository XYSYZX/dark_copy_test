#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
//#include "cuda_runtime.h"
//#include "curand.h"
//#include "cublas_v2.h"
#include "darknet.h"
/*
extern "C"{
#include "cuda.h"
}
*/

void get_topk_gpu(float *x_gpu, int length, int *y_idx, int topk)
{
    abs_gpu(x_gpu, x_gpu, length);

    thrust::device_vector<int> x_dev_idx(length);
    thrust::host_vector<int> x_dev_idx(length);
    //thrust::device_ptr<int> x_dev_idx = thrust::device_malloc<int>(length);
    thrust::sequence(x_dev_idx.begin(), x_dev_idx.end());

    thrust::device_ptr<float> x_dev(x_gpu);
    thrust::sort_by_key(thrust::device, x_dev, x_dev + length, x_dev_idx, thrust::greater<float>());
    //thrust::copy(x_dev, x_dev + topk, y);
    //int *x_idx = thrust::raw_pointer_cast(x_dev_idx);
    //x_gpu = thrust::raw_pointer_cast(x_dev);
    cuda_pull_int_array(x_idx, y_idx, topk);
    //thrust::device_free(x_dev_idx);
    x_dev_idx.clear();
    x_dev_idx.shrink_to_fit();
}

