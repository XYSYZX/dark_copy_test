#include "bound_adjust.h"
#include "darknet.h"
#include <stdlib.h>
#include <stdio.h>

void get_weight_bound(layer l)
{
#ifdef GPU
    get_weight_bound_gpu(l);
    return;
#endif
    int num = l.c/l.groups*l.n*l.size*l.size;
    float min = 3.402823466e+38F;
    float max = 1.175494351e-38F;
    for(int i=0; i<num; i++){
        if(max < l.weights[i]) max = l.weights[i];
        if(min > l.weights[i]) min = l.weights[i];
    }
    if(*l.max_weight < max)  *l.max_weight = max;
    if(*l.min_weight > min)  *l.min_weight = min;
    printf("max: %.5f, min: %.5f\n", *l.max_weight, *l.min_weight);
}

void get_output_bound(layer l)
{
#ifdef GPU
    get_output_bound_gpu(l);
    return;
#endif
    int num = l.outputs*l.batch;
    float min = 3.402823466e+38F;
    float max = 1.175494351e-38F;
    for(int i=0; i<num; i++){
        if(max < l.output[i]) max = l.output[i];
        if(min > l.output[i]) min = l.output[i];
    }
    if(*l.max_output < max) *l.max_output = max;
    if(*l.min_output > min) *l.min_output = min;
    printf("layer indexc: %d, max: %.5f, min: %.5f\n", l.index, *l.max_output, *l.min_output);
}   

void constrain_cpu(float *x, int max, int min, int nums)
{
    for(int i=0; i<nums; i++){
        if(x[i] > max) x[i] = max;
        if(x[i] > min) x[i] = min;
    }
}

void save_weight_bound(network *net, char *file)
{
    FILE *fp = 0;
    fp = fopen(file, "w");
    for(int i = 0; i < net->n; i++){
        if(net->layers[i].type == CONVOLUTIONAL){
            fprintf(fp, "%f %f\n", *(net->layers[i].max_weight), *(net->layers[i].min_weight));
        }
    }
    fclose(fp);
}
        

void save_output_bound(network *net, char *file)
{
    FILE *fp = 0;
    fp = fopen(file, "w");
    for(int i = 0; i < net->n; i++){
        if(net->layers[i].type == CONVOLUTIONAL)
            fprintf(fp, "%f %f\n", *(net->layers[i].max_output), *(net->layers[i].min_output));
        if(net->layers[i].type == MAXPOOL)
            fprintf(fp, "%f %f\n", *(net->layers[i].max_output), *(net->layers[i].min_output));
        if(net->layers[i].type == UPSAMPLE)
            fprintf(fp, "%f %f\n", *(net->layers[i].max_output), *(net->layers[i].min_output));
        if(net->layers[i].type == ROUTE)
            fprintf(fp, "%f %f\n", *(net->layers[i].max_output), *(net->layers[i].min_output));
        if(net->layers[i].type == SHORTCUT)
            fprintf(fp, "%f %f\n", *(net->layers[i].max_output), *(net->layers[i].min_output));
    }
    fclose(fp);
}

void load_weight_bound(network *net, char *file)
{
    FILE *fp = NULL;
    fp = fopen(file, "r");
    for(int i = 0; i < net->n; i++){
        if(net->layers[i].type == CONVOLUTIONAL){
            fscanf(fp, "%f %f\n", net->layers[i].max_weight, net->layers[i].min_weight);
	    *net->layers[i].max_weight += 0.000001;
	    *net->layers[i].min_weight -= 0.000001;
            fprintf(stderr, "max w: %f, min w: %f\n", *net->layers[i].max_weight, *net->layers[i].min_weight);
        }
    }
    fclose(fp);
}

void load_output_bound(network *net, char *file)
{
    FILE *fp = 0;
    fp = fopen(file, "r");
    for(int i = 0; i < net->n; i++){
        LAYER_TYPE ltype = net->layers[i].type;
        if(ltype == CONVOLUTIONAL || ltype == MAXPOOL || ltype == ROUTE || ltype == SHORTCUT || ltype == UPSAMPLE){
            fscanf(fp, "%f %f\n", net->layers[i].max_output, net->layers[i].min_output);
            fprintf(stderr, "max o: %f, min o: %f\n", *net->layers[i].max_output, *net->layers[i].min_output);
        }
    }
    fclose(fp);
}

void load_output_bound_2(network *net, char *file)
{
    FILE *fp = 0;
    fp = fopen(file, "r");
    float max_f;
    float min_f;
    for(int i = 0; i < net->n; i++){
        LAYER_TYPE ltype = net->layers[i].type;
        if(ltype == CONVOLUTIONAL || ltype == MAXPOOL || ltype == ROUTE || ltype == SHORTCUT || ltype == UPSAMPLE){
            fscanf(fp, "%f %f\n", &max_f, &min_f);
            *net->layers[i].max_output = max_f / 2;
            *net->layers[i].min_output = min_f / 2;
            printf("max w: %.5f, min w: %.5f\n", *net->layers[i].max_output, *net->layers[i].min_output);
        }
    }
    fclose(fp);
}

void load_output_bound_8(network *net, char *file)
{
    FILE *fp = 0;
    fp = fopen(file, "r");
    float max_f;
    float min_f;
    for(int i = 0; i < net->n; i++){
        LAYER_TYPE ltype = net->layers[i].type;
        if(ltype == CONVOLUTIONAL || ltype == MAXPOOL || ltype == ROUTE || ltype == SHORTCUT || ltype == UPSAMPLE){
            fscanf(fp, "%f %f\n", &max_f, &min_f);
            *net->layers[i].max_output = max_f / 8;
            *net->layers[i].min_output = min_f;
            printf("max w: %.5f, min w: %.5f\n", *net->layers[i].max_output, *net->layers[i].min_output);
        }
    }
    fclose(fp);
}

void check_weights(network *net)
{
#ifdef GPU
    check_weights_gpu(net);
    return;
#endif
    for(int i = 0; i < net->n; i++){
        if(net->layers[i].type == CONVOLUTIONAL){
            layer l = net->layers[i];
            int num = l.c/l.groups*l.n*l.size*l.size;
            float max = *l.max_weight;
            float min = *l.min_weight;
            for(int j = 0; j < num; j++){
                if(l.weights[i] < min) l.weights[i] = min;
                if(l.weights[i] > max) l.weights[i] = max;
            }
        }
    }
}

void check_outputs(layer l)
{
    int num = l.outputs*l.batch;
    float max = *l.max_output;
    float min = *l.min_output;
    for(int i = 0; i < num; i++){
        if(l.output[i] < min) l.output[i] = min;
        if(l.output[i] > max) l.output[i] = max;
    }
}


