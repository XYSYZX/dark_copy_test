#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "im2col.h"
#include "cuda.h"
}

// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

__global__ void im2col_gpu_kernel(const int n, const float* data_im, //n is num of windows that kernel pass by, e.g. n=4*4*1, height_col=width_col=4, channel=1
        const int height, const int width, const int ksize, //height, width is size of image(assume 4),ksize is size of kernel e.g.=3
        const int pad, //assume pad=1
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;  //every single thread process a window, resposible for a column in data_col
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col; //assume it's the sixth window(size 3*3), index=5, w_out=5%4=1,
        int h_index = index / width_col;   //h_index=5/4=1
        int h_out = h_index % height_col;   //h_out=1%4=1
        int channel_in = h_index / height_col;  //the index of channel now, channel_in=1/4=0
        int channel_out = channel_in * ksize * ksize;     //channel_out=0*3*3=0
        int h_in = h_out * stride - pad;  //h_in=1*1-1=0
        int w_in = w_out * stride - pad;  //w_in=1*1-1=0
        float* data_col_ptr = data_col;   
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;  //data_col_ptr point at (++(0*4+1)*4+1=5), meaning point at 5th column in data_col_ptr
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;  //data_im_ptr point at (++(0*4+0)*4+0=0), meaning the first element(left-up) in a window
        for (int i = 0; i < ksize; ++i) {   //assume every kernel has 3*3 elements, every one get its related value in image, making it to a colume
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;  //h=0+0
                int w = w_in + j;  //w=0+0

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;  //give *data_col_ptr the value of first element in sixth window(3*3)

                //*data_col_ptr = data_im_ptr[ii * width + jj];

                data_col_ptr += height_col * width_col;  //move to next row(++4*4)
            }
        }
    }
}

void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;  
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
}
