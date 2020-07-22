#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)  //batch: num of images in a batch
{
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;  //激活函数层一张输入图片中所有元素个数
    l.outputs = inputs; //激活函数层对应一张输入图片的输出元素个数，激活函数层不改变输入输出的个数
    l.batch=batch;      //一个batch中图片的张数

    l.output = calloc(batch*inputs, sizeof(float*));  // 激活函数层的所有输出（包含整个batch的）
    l.delta = calloc(batch*inputs, sizeof(float*));   // 激活函数层的误差损失项 （包含整个batch的）
 
    // 激活函数层的前向,反向传播函数
    l.forward = forward_activation_layer;  //make forward function(in darknet.h) become forward_activation_layer
    l.backward = backward_activation_layer;
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}

void forward_activation_layer(layer l, network net)
{
    // l.output = net.input
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    // 计算输入经过激活函数的输出值
    activate_array(l.output, l.outputs*l.batch, l.activation);
}


void backward_activation_layer(layer l, network net)
{
    // 计算激活函数对加权输入的导数，并乘以delta，得到当前层最终的误差项delta
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    // net.delta = l.delta 误差项向前一层传播，存入中转位置 net.delta
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_activation_layer_gpu(layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
