#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

/*
**  根据输入的激活函数名称，返回标准（darknet定义的枚举类型）的激活函数类别
**  输入：s    C风格字符数组，激活函数名称，比如relu,logistic等等
**  返回：ACTIVATION   激活函数类别，枚举类型
**  说明：该函数仅仅通过匹配字符数组，返回标准的激活函数类别而已；
**       如果输入的激活函数名称未能识别，则统一使用RELU
*/

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu")==0) return SELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

/*
** 根据不同的激活函数类型，调用不同的激活函数处理单个输入元素x
** 输入： x    待处理的元素（单个）
**       a    激活函数类型
*/

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LOGGY:
            return loggy_activate(x);
        case RELU:
            return relu_activate(x);
        case ELU:
            return elu_activate(x);
        case SELU:
            return selu_activate(x);
        case RELIE:
            return relie_activate(x);
        case RAMP:
            return ramp_activate(x);
        case LEAKY:
            return leaky_activate(x);
        case TANH:
            return tanh_activate(x);
        case PLSE:
            return plse_activate(x);
        case STAIR:
            return stair_activate(x);
        case HARDTAN:
            return hardtan_activate(x);
        case LHTAN:
            return lhtan_activate(x);
    }
    return 0;
}

/**
** 用激活函数处理输入x中的每一个元素
** 输入： x    待处理的数组，一般为网络层每个神经元的加权输入Wx+b，在本函数中也相当于是输出（本地操作～）
**       n    x中含有多少个元素
**       a    激活函数类型
** 说明：该函数会逐个处理x中的元素，注意是逐个；该函数一般用于每一层网络的前向传播函数中，比如forward_connected_layer()等，
**      用在最后一步，该函数的输出即为每一层网络的输出
*/
void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    // 逐个处理x中的元素
    for(i = 0; i < n; ++i){
	// 根据不同的激活函数类型，调用不同的激活函数处理
        x[i] = activate(x[i], a);
    }
}

/*
** 根据不同的激活函数求取对输入的梯度（导数）
** 输入： x    激活函数接收的输入值
**       a    激活函数类型，包括的激活函数类型见activations.h中枚举类型ACTIVATION的定义
** 输出： 激活函数关于输入x的导数值
*/

float gradient(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient(x);
        case LOGISTIC:
            return logistic_gradient(x);
        case LOGGY:
            return loggy_gradient(x);
        case RELU:
            return relu_gradient(x);
        case ELU:
            return elu_gradient(x);
        case SELU:
            return selu_gradient(x);
        case RELIE:
            return relie_gradient(x);
        case RAMP:
            return ramp_gradient(x);
        case LEAKY:
            return leaky_gradient(x);
        case TANH:
            return tanh_gradient(x);
        case PLSE:
            return plse_gradient(x);
        case STAIR:
            return stair_gradient(x);
        case HARDTAN:
            return hardtan_gradient(x);
        case LHTAN:
            return lhtan_gradient(x);
    }
    return 0;
}

/*
** 计算激活函数对加权输入的导数，计算delta
** x: l.output（维度为l.batch * l.out_c * l.out_w * l.out_h）
** n: l.output的维度，l.batch * l.out_c * l.out_w * l.out_h
** ACTIVATION:  激活函数类型, 大部分激活函数求导可用l.output计算
*/

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);  //the error of a layer, every elements multiply
    }
} 

