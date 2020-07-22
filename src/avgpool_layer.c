#include "avgpool_layer.h"
#include "cuda.h"
#include <stdio.h>

/**
 * 构建平均池化层
 * @param batch 一个batch包含图片的张数
 * @param w 输入图片的宽度
 * @param h 输入图片的高度
 * @param c 输入图片的通道数
 * @return
 */
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l = {0};
    l.type = AVGPOOL;    //type of layer
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;     // 平均池化后宽度为1, 这里only计算输入图像的一块，在进入平均池化层已经做了划分，所有这里的输出宽度为1
    l.out_h = 1;      //平均池化后高度为1
    l.out_c = c;     // 输出图片的通道数
    l.outputs = l.out_c;  // 平均池化层对应一张输入图片的输出元素个数
    l.inputs = h*w*c;     // 平均池化层一张输入图片中所有元素个数
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float)); // 平均池化层的所有输出（包含整个batch）
    l.delta =   calloc(output_size, sizeof(float));  // 平均池化层的误差项（包含整个batch）

    // 平均池化层的前向,反向传播函数
    l.forward = forward_avgpool_layer;
    l.backward = backward_avgpool_layer;
    #ifdef GPU
    l.forward_gpu = forward_avgpool_layer_gpu;
    l.backward_gpu = backward_avgpool_layer_gpu;
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

/**
 * 平均池化层前向出传播
 * @param l 当前平均池化层
 * @param net 整个网络
 */
void forward_avgpool_layer(const avgpool_layer l, network net)  //why all image(h*w) become 1 pixel?
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){  // 统计每个batch中每张图片, batch is num of images in a batch
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){   // 统计每一张图片
                int in_index = i + l.h*l.w*(k + b*l.c);
                l.output[out_index] += net.input[in_index]; // 计算每一张图片的总和
            }
            l.output[out_index] /= l.h*l.w;   //平均池化运算
        }
    }
}

/**
 * 平均池化层的反向传播
 * @param l 当前平均池化层
 * @param net 整个网络
 */
void backward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            int out_index = k + b*l.c;
            for(i = 0; i < l.h*l.w; ++i){
                int in_index = i + l.h*l.w*(k + b*l.c);
		// 下一层的误差项的值会平均分配到上一层对应区块中的所有神经元
                net.delta[in_index] += l.delta[out_index] / (l.h*l.w);
            }
        }
    }
}

