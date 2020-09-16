#include "bit_attack.h"
//#include <stdlib.h>
//#include <stdio.h>


float **make_2d_array_float(int n, int m)
{
    float **x = (float **)calloc(n, sizeof(float*));
    for(int i = 0; i < n; i++){
        x[i] = (float *)calloc(m, sizeof(float));
    }
    return x;
}

int **make_2d_array_int(int n, int m)
{
    int **x = (int **)calloc(n, sizeof(int*));
    for(int i = 0; i < n; i++){
        x[i] = (int *)calloc(m, sizeof(int));
    }
    return x;
}

void free_2d_array_float(float **x, int n)
{
    for(int i = 0; i < n; i++) free(x[i]);
    free(x);
}

void free_2d_array_int(int **x, int n)
{
    for(int i = 0; i < n; i++) free(x[i]);
    free(x);
}

void zero_2d_array_float(float **x, int n, int m)
{
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            x[i][j] = 0;
}

void zero_2d_array_int(int **x, int n, int m)
{
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            x[i][j] = 0;
}

void print_2d_array_int(int **x, int m, int n)
{
    int i, j;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++)
            printf("%d ", x[i][j]);
        printf("\n");
    }
}

void print_2d_array_float(float **x, int m, int n)
{
    int i, j;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++)
            printf("%f ", x[i][j]);
        printf("\n");
    }
}

void set_attack_args(network *net)
{
    int i;
    int topk;
    attack_args *a = net->attack;
    int layer_num = net->n;
    int iter = a->iter;
    a->layer_num = layer_num;
    
    //input
    topk = a->topk_inputs;
    a->grads_loc_inputs = make_2d_array_int(1, topk * iter);
    a->mloss_loc_inputs = make_2d_array_int(1, topk);
    a->mloss_inputs = make_2d_array_float(1, topk);
    a->avf_inputs = make_2d_array_float(1, topk);

    a->inputs_len = (int *)calloc(1, sizeof(int));
    a->inputs_len[0] = net->inputs*net->batch;

    a->input_grads = (float **)calloc(1, sizeof(float*));
    a->input_grads_gpu = (float **)calloc(1, sizeof(float*));
    a->input_grads[0] = net->delta;
    a->input_grads_gpu[0] = net->delta_gpu;

    a->inputs = (float **)calloc(1, sizeof(float*));
    a->inputs_gpu = (float **)calloc(1, sizeof(float*));
    a->inputs[0] = net->input;
    a->inputs_gpu[0] = net->input_gpu;

    //weight
    topk = a->topk_weights;
    a->grads_loc_weights = make_2d_array_int(layer_num, topk * iter);
    a->mloss_loc_weights = make_2d_array_int(layer_num, topk);
    a->mloss_weights = make_2d_array_float(layer_num, topk);
    a->avf_weights = make_2d_array_float(layer_num, topk);

    a->weights_len = (int *)calloc(net->n, sizeof(int));
    for(i = 0; i < net->n; i++) a->weights_len[i] = net->layers[i].nweights;

    a->weight_grads = (float **)calloc(net->n, sizeof(float*));
    a->weight_grads_gpu = (float **)calloc(net->n, sizeof(float*));
    for(i = 0; i < net->n; i++) a->weight_grads[i] = net->layers[i].weight_updates;
    for(i = 0; i < net->n; i++) a->weight_grads_gpu[i] = net->layers[i].weight_updates_gpu;

    a->weights = (float **)calloc(net->n, sizeof(float*));
    a->weights_gpu = (float **)calloc(net->n, sizeof(float*));
    for(i = 0; i < net->n; i++) a->weights[i] = net->layers[i].weights;
    for(i = 0; i < net->n; i++) a->weights_gpu[i] = net->layers[i].weights_gpu;

    //bias
    topk = a->topk_biases;
    a->grads_loc_biases = make_2d_array_int(layer_num, topk * iter);
    a->mloss_loc_biases = make_2d_array_int(layer_num, topk);
    a->mloss_biases = make_2d_array_float(layer_num, topk);
    a->avf_biases = make_2d_array_float(layer_num, topk);

    a->biases_len = (int *)calloc(net->n, sizeof(int));
    for(i = 0; i < net->n; i++) a->biases_len[i] = net->layers[i].nbiases;

    a->bias_grads = (float **)calloc(net->n, sizeof(float*));
    a->bias_grads_gpu = (float **)calloc(net->n, sizeof(float*));
    for(i = 0; i < net->n; i++) a->bias_grads[i] = net->layers[i].bias_updates;
    for(i = 0; i < net->n; i++) a->bias_grads_gpu[i] = net->layers[i].bias_updates_gpu;

    a->biases = (float **)calloc(net->n, sizeof(float*));
    a->biases_gpu = (float **)calloc(net->n, sizeof(float*));
    for(i = 0; i < net->n; i++) a->biases[i] = net->layers[i].biases;
    for(i = 0; i < net->n; i++) a->biases_gpu[i] = net->layers[i].biases_gpu;
    
    //output
    topk = a->topk_outputs;
    a->grads_loc_outputs = make_2d_array_int(layer_num, topk * iter);
    a->mloss_loc_outputs = make_2d_array_int(layer_num, topk);
    a->mloss_outputs = make_2d_array_float(layer_num, topk);
    a->avf_outputs = make_2d_array_float(layer_num, topk);

    a->outputs_len = (int *)calloc(net->n, sizeof(int));
    for(i = 0; i < net->n; i++) a->outputs_len[i] = net->layers[i].outputs*net->batch;

    a->output_grads = (float **)calloc(net->n, sizeof(float*));
    a->output_grads_gpu = (float **)calloc(net->n, sizeof(float*));
    for(i = 0; i < net->n; i++) a->output_grads[i] = net->layers[i].delta;
    for(i = 0; i < net->n; i++) a->output_grads_gpu[i] = net->layers[i].delta_gpu;

    a->outputs = (float **)calloc(net->n, sizeof(float*));
    a->outputs_gpu = (float **)calloc(net->n, sizeof(float*));
    for(i = 0; i < net->n; i++) a->outputs[i] = net->layers[i].output;
    for(i = 0; i < net->n; i++) a->outputs_gpu[i] = net->layers[i].output_gpu;
}

void free_attack_args(attack_args a)
{
    int layer_num = a.layer_num;
    {
        if(a.grads_loc_inputs) free_2d_array_int(a.grads_loc_inputs, 1);
        if(a.mloss_loc_inputs) free_2d_array_int(a.mloss_loc_inputs, 1);
        if(a.mloss_inputs) free_2d_array_float(a.mloss_inputs, 1);
        if(a.avf_inputs) free_2d_array_float(a.avf_inputs, 1);
        if(a.inputs_len) free(a.inputs_len);
        if(a.input_grads) free(a.input_grads);
        if(a.input_grads_gpu) free(a.input_grads_gpu);
        if(a.inputs) free(a.inputs);
        if(a.inputs_gpu) free(a.inputs_gpu);
    }
    {
        if(a.grads_loc_weights) free_2d_array_int(a.grads_loc_weights, layer_num);
        if(a.mloss_loc_weights) free_2d_array_int(a.mloss_loc_weights, layer_num);
        if(a.mloss_weights) free_2d_array_float(a.mloss_weights, layer_num);
        if(a.avf_weights) free_2d_array_float(a.avf_weights, layer_num);
        if(a.weights_len) free(a.weights_len);
        if(a.weight_grads) free(a.weight_grads);
        if(a.weight_grads_gpu) free(a.weight_grads_gpu);
        if(a.weights) free(a.weights);
        if(a.weights_gpu) free(a.weights_gpu);
    }
    {
        if(a.grads_loc_biases) free_2d_array_int(a.grads_loc_biases, layer_num);
        if(a.mloss_loc_biases) free_2d_array_int(a.mloss_loc_biases, layer_num);
        if(a.mloss_biases) free_2d_array_float(a.mloss_biases, layer_num);
        if(a.avf_biases) free_2d_array_float(a.avf_biases, layer_num);
        if(a.biases_len) free(a.biases_len);
        if(a.bias_grads) free(a.bias_grads);
        if(a.bias_grads_gpu) free(a.bias_grads_gpu);
        if(a.biases) free(a.biases);
        if(a.biases_gpu) free(a.biases_gpu);
    }
    {
        if(a.grads_loc_outputs) free_2d_array_int(a.grads_loc_outputs, layer_num);
        if(a.mloss_loc_outputs) free_2d_array_int(a.mloss_loc_outputs, layer_num);
        if(a.mloss_outputs) free_2d_array_float(a.mloss_outputs, layer_num);
        if(a.avf_outputs) free_2d_array_float(a.avf_outputs, layer_num);
        if(a.outputs_len) free(a.outputs_len);
        if(a.output_grads) free(a.output_grads);
        if(a.output_grads_gpu) free(a.output_grads_gpu);
        if(a.outputs) free(a.outputs);
        if(a.outputs_gpu) free(a.outputs_gpu);
    }
}

void attack_data(network *net, load_args args, load_args val_args)
{
    float loss = 0;
    int i, j;
    int type;

    attack_args *attack = net->attack;
    int iter = args.m / args.n;
    attack->iter = iter;
    set_attack_args(net);


    data val, buffer;
    args.d = &buffer;

    pthread_t load_thread = load_data(args);
    attack->seen_img += attack->n;

    for(i = 0; i < iter; i++){
        attack->iter_idx = i;
        pthread_join(load_thread, 0);
        val = buffer;
        fprintf(stderr, "images: %d\n", attack->seen_img);
        loss += network_predict_search(net, val);
        for(j = 0; j < attack->layer_num; j++){
            //printf("grad: %x, gpu: %x\n", attack->grads[j], attack->grads_gpu[j]);
            attack->layer_idx = j;
            get_topk_grad(attack);
        }
        load_thread = load_data(args);
        attack->seen_img += args.n;
    }
    loss /= iter;
    get_max_loss(attack);

    attack->seen_img = 0;
    for(i = 0; i < 4; i++){
        type = i;
        get_avf(net, val_args, type);
    }

    free_attack_args(*attack);
}

void get_topk_grad(attack_args *a)
{
    int j = a->layer_idx;
    int i = a->iter_idx;
    if(j == 0){
        if(a->input_grads[j] && a->input_grads_gpu[j]){
            get_topk(a->input_grads[j], a->input_grads_gpu[j], a->inputs_len[j], a->grads_loc_inputs[j]+i*a->topk_inputs, a->topk_inputs);
        }
    }
    if(a->weight_grads[j] && a->weight_grads_gpu[j]){
        get_topk(a->weight_grads[j], a->weight_grads_gpu[j], a->weights_len[j], a->grads_loc_weights[j]+i*a->topk_weights, a->topk_weights);
    }
    if(a->bias_grads[j] && a->bias_grads_gpu[j]){
        get_topk(a->bias_grads[j], a->bias_grads_gpu[j], a->biases_len[j], a->grads_loc_biases[j]+i*a->topk_biases, a->topk_biases);
    }
    if(a->output_grads[j] && a->output_grads_gpu[j]){
        get_topk(a->output_grads[j], a->output_grads_gpu[j], a->outputs_len[j], a->grads_loc_outputs[j]+i*a->topk_outputs, a->topk_outputs);
    }
}


void progressive_attack(network *net)
{
    
    attack_args a = *(net->attack);
    if(a.sign_attack){
        if(a.a_input){
        #ifdef GPU
            cuda_pull_array(a.grads_gpu[a.layer_idx], a.grads[a.layer_idx], a.len[a.layer_idx]);
        #endif
            sign_attacker(a.x[a.layer_idx], a.mloss_loc[a.layer_idx], a.topk, a.grads[a.layer_idx], a.epsilon);
        }
        else{
            if(a.reverse == 1){
            #ifdef GPU
                sign_attacker_gpu(a.x_gpu[a.layer_idx], a.mloss_loc[a.layer_idx], a.topk, a.grads_gpu[a.layer_idx], a.epsilon);
            #else
                sign_attacker(a.x[a.layer_idx], a.mloss_loc[a.layer_idx], a.topk, a.grads[a.layer_idx], a.epsilon);
            #endif
            }
            else{
            #ifdef GPU
                sign_delete_gpu(a.x_gpu[a.layer_idx], a.mloss_loc[a.layer_idx], a.topk, a.grads_gpu[a.layer_idx], a.epsilon);
            #else
                sign_delete(a.x[a.layer_idx], a.mloss_loc[a.layer_idx], a.topk, a.grads[a.layer_idx], a.epsilon);
            #endif
            }
        }
    }
    else{
        if(a.a_input){
            for(int i = 0; i < a.topk; i++){
                inject_noise_float_manybit(a.x[a.layer_idx], a.mloss_loc[a.layer_idx][i], a.fb_len, a.flipped_bit);
            }
        }
        else{
            for(int i = 0; i < a.topk; i++){
            #ifdef GPU
                inject_noise_float_manybit_gpu(a.x_gpu[a.layer_idx], a.mloss_loc[a.layer_idx][i], a.fb_len, a.flipped_bit);
            #else
                inject_noise_float_manybit(a.x[a.layer_idx], a.mloss_loc[a.layer_idx][i], a.fb_len, a.flipped_bit);
            #endif
            }
        }
    }
    if(a.a_weight || a.a_bias) {
        net->attack->reverse *= -1;
    }
}

void single_attack(network *net)
{
    attack_args a = *(net->attack);
    int offset = a.k_idx;
    if(a.sign_attack){
        if(a.a_input){
        #ifdef GPU
            if(a.k_idx == 0) cuda_pull_array(a.grads_gpu[a.layer_idx], a.grads[a.layer_idx], a.len[a.layer_idx]);
        #endif
            sign_attacker(a.x[a.layer_idx], a.mloss_loc[a.layer_idx]+offset, 1, a.grads[a.layer_idx], a.epsilon);
        }
        else{
            if(a.reverse == 1){
            #ifdef GPU
                sign_attacker_gpu(a.x_gpu[a.layer_idx], a.mloss_loc[a.layer_idx]+offset, 1, a.grads_gpu[a.layer_idx], a.epsilon);
            #else
                sign_attacker(a.x[a.layer_idx], a.mloss_loc[a.layer_idx]+offset, 1, a.grads[a.layer_idx], a.epsilon);
            #endif
            }
            else{
            #ifdef GPU
                sign_delete_gpu(a.x_gpu[a.layer_idx], a.mloss_loc[a.layer_idx]+offset, 1, a.grads_gpu[a.layer_idx], a.epsilon);
            #else
                sign_delete(a.x[a.layer_idx], a.mloss_loc[a.layer_idx]+offset, 1, a.grads[a.layer_idx], a.epsilon);
            #endif
            }
        }
 
    }
    else{
        if(a.a_input){
            inject_noise_float_manybit(a.x[a.layer_idx], a.mloss_loc[a.layer_idx][offset], a.fb_len, a.flipped_bit);
        }
        else{
        #ifdef GPU
            inject_noise_float_manybit_gpu(a.x_gpu[a.layer_idx], a.mloss_loc[a.layer_idx][offset], a.fb_len, a.flipped_bit);
        #else
            inject_noise_float_manybit(a.x[a.layer_idx], a.mloss_loc[a.layer_idx][offset], a.fb_len, a.flipped_bit);
        #endif
        }
    }
}

float sign(float x)
{
    float y;
    if(x > .000001) y = 1;
    else if(x < -0.000001) y = -1;
    else y = 0;
    return y;
}

void sign_attacker(float *x, int *loc, int topk, float *grad, float epsilon)
{
    int i, idx;
    //printf("attack sign!\n");
    for(i = 0; i < topk; i++){
        idx = loc[i];
        float x_sign = sign(grad[idx]);
        x[idx] -= epsilon * x_sign;
    }
}

void sign_delete(float *x, int *loc, int topk, float *grad, float epsilon)
{
    int i, idx;
    //printf("delete sign!\n");
    for(i = 0; i < topk; i++){
        idx = loc[i];
        float x_sign = sign(grad[idx]);
        x[idx] += epsilon * x_sign;
    }
}

void get_max_loss(attack_args *a)
{
    int i, j;
    int layer_num = a->layer_num;
    int iter = a->iter;
    int **mloss_freq_input = make_2d_array_int(1, a->topk_inputs);
    int **mloss_freq_weight = make_2d_array_int(layer_num, a->topk_weights);
    int **mloss_freq_bias = make_2d_array_int(layer_num, a->topk_biases);
    int **mloss_freq_output = make_2d_array_int(layer_num, a->topk_outputs);
    int *cal_input;
    int *cal_weight;
    int *cal_bias;
    int *cal_output;

    for(i = 0; i < layer_num; i++){
        if(i == 0){
            cal_input = (int *)calloc(a->inputs_len[i], sizeof(int));
            for(j = 0; j < iter * a->topk_inputs ; j++){
                cal_input[a->grads_loc_inputs[i][j]] += 1;
            }
            top_k_int(cal_input, a->inputs_len[i], a->topk_inputs, a->mloss_loc_inputs[i], mloss_freq_input[i]);
            free(cal_input);
        }
        if(a->weights_len[i]){
            cal_weight = (int *)calloc(a->weights_len[i], sizeof(int));
            for(j = 0; j < iter * a->topk_weights ; j++){
                cal_weight[a->grads_loc_weights[i][j]] += 1;
            }
            top_k_int(cal_weight, a->weights_len[i], a->topk_weights, a->mloss_loc_weights[i], mloss_freq_weight[i]);
            free(cal_weight);
        }
        if(a->biases_len[i]){
            cal_bias = (int *)calloc(a->biases_len[i], sizeof(int));
            for(j = 0; j < iter * a->topk_biases ; j++){
                cal_bias[a->grads_loc_biases[i][j]] += 1;
            }
            top_k_int(cal_bias, a->biases_len[i], a->topk_biases, a->mloss_loc_biases[i], mloss_freq_bias[i]);
            free(cal_bias);
        }
        if(a->outputs_len[i]){
            cal_output = (int *)calloc(a->outputs_len[i], sizeof(int));
            for(j = 0; j < iter * a->topk_outputs ; j++){
                cal_output[a->grads_loc_outputs[i][j]] += 1;
            }
            top_k_int(cal_output, a->outputs_len[i], a->topk_outputs, a->mloss_loc_outputs[i], mloss_freq_output[i]);
            free(cal_output);
        }
    }
    printf("input: \n");
    print_2d_array_int(a->mloss_loc_inputs, 1, a->topk_inputs);
    print_2d_array_int(mloss_freq_input, 1, a->topk_inputs);
    printf("weight: \n");
    print_2d_array_int(a->mloss_loc_weights, layer_num, a->topk_weights);
    print_2d_array_int(mloss_freq_weight, layer_num, a->topk_weights);
    printf("bias: \n");
    print_2d_array_int(a->mloss_loc_biases, layer_num, a->topk_biases);
    print_2d_array_int(mloss_freq_bias, layer_num, a->topk_biases);
    printf("output: \n");
    print_2d_array_int(a->mloss_loc_outputs, layer_num, a->topk_outputs);
    print_2d_array_int(mloss_freq_output, layer_num, a->topk_outputs);
    free_2d_array_int(mloss_freq_input, 1);
    free_2d_array_int(mloss_freq_weight, layer_num);
    free_2d_array_int(mloss_freq_bias, layer_num);
    free_2d_array_int(mloss_freq_output, layer_num);
}

void get_avf(network *net, load_args args, int type)
{
    attack_args *a = net->attack;
    float avg_loss = 0;
    int layer_num = net->n;

    int i, j, k;
    if(a->sign_attack) args.n = net->batch;
    int iter = args.m / args.n;
    a->iter = iter;
    data val, buffer;
    args.d = &buffer;

    switch(type){
        case 0:
            a->a_input = 1;
            a->layer_num = 1;
            a->grads = a->input_grads;
            a->grads_gpu = a->input_grads_gpu;
            a->len = a->inputs_len;
            a->x = a->inputs;
            a->x_gpu = a->inputs_gpu;
            a->topk = a->topk_inputs;
            a->mloss = a->mloss_inputs;
            a->mloss_loc = a->mloss_loc_inputs;
            a->avf = a->avf_inputs;
            break;
        case 1:
            a->a_weight = 1;
            a->layer_num = layer_num;
            a->grads = a->weight_grads;
            a->grads_gpu = a->weight_grads_gpu;
            a->len = a->weights_len;
            a->x = a->weights;
            a->x_gpu = a->weights_gpu;
            a->topk = a->topk_weights;
            a->mloss = a->mloss_weights;
            a->mloss_loc = a->mloss_loc_weights;
            a->avf = a->avf_weights;
            break;
        case 2:
            a->a_bias = 1;
            a->layer_num = layer_num;
            a->grads = a->bias_grads;
            a->grads_gpu = a->bias_grads_gpu;
            a->len = a->biases_len;
            a->x = a->biases;
            a->x_gpu = a->biases_gpu;
            a->topk = a->topk_biases;
            a->mloss = a->mloss_biases;
            a->mloss_loc = a->mloss_loc_biases;
            a->avf = a->avf_biases;
            break;
        case 3:
            a->a_output = 1;
            a->layer_num = layer_num;
            a->grads = a->output_grads;
            a->grads_gpu = a->output_grads_gpu;
            a->len = a->outputs_len;
            a->x = a->outputs;
            a->x_gpu = a->outputs_gpu;
            a->topk = a->topk_outputs;
            a->mloss = a->mloss_outputs;
            a->mloss_loc = a->mloss_loc_outputs;
            a->avf = a->avf_outputs;
            break;
        default:
            printf("wrong attack type!\n");
            return;
    }

    pthread_t load_thread = load_data(args);
    for(i = 0; i < iter; i++){
        a->iter_idx = i;
        pthread_join(load_thread, 0);
        fprintf(stderr, "images: %d\n", a->seen_img);
        val = buffer;
        avg_loss += network_predict_search(net, val);
        for(j = 0; j < a->layer_num; j++){
            a->layer_idx = j;
            if(a->len[j] == 0) continue;
            if(a->progress_attack){
                if(a->a_weight || a->a_bias){
                    progressive_attack(net);
                }

                a->mloss[j][0] += network_predict_attack(net, val);

                if(a->a_weight || a->a_bias){
                    progressive_attack(net);
                }
            }
            else{
                for(k = 0; k < a->topk; k++){
                    a->k_idx = k;
                    if(a->a_weight || a->a_bias){
                        single_attack(net);
                    }
                    a->mloss[j][k] += network_predict_attack(net, val);
                    if(a->a_weight || a->a_bias){
                        single_attack(net);
                    }
                    //printf("single: %f\n", attack->mloss[j][i*attack->topk + k]);
                }
            }
        }
        load_thread = load_data(args);
        a->seen_img += args.n;
    }

    avg_loss /= iter;
    for(i = 0; i < a->layer_num; i++){
        for(j = 0; j < a->topk; j++){
            a->mloss[i][j] /= iter;
        }
    }
    cal_avf(a, avg_loss);
    printf("type: %d, avf: \n", type);
    print_2d_array_float(a->avf, a->layer_num, a->topk);
    if(type == 0) a->a_input = 0;
    if(type == 1) a->a_weight = 0;
    if(type == 2) a->a_bias = 0;
    if(type == 3) a->a_output = 0;
    a->seen_img = 0;
}

void cal_avf(attack_args *a, float avg_loss)
{
    int i, j;
    float max = 0;
    float min = avg_loss;
    float **avf = a->avf;
    float **mloss = a->mloss;
    for(i = 0; i < a->layer_num; i++){
        for(j = 0; j < a->topk; j++){
            max = mloss[i][j] > max ? mloss[i][j]: max;
            if(mloss[i][j] > 0 && mloss[i][j] < min){
                printf("min is bigger!\n");
                printf("layer: %i, min: %f, this: %f\n", i, min, mloss[i][j]);
                //return;
            }
        }
    }
    float factor = max - min;
    for(i = 0; i < a->layer_num; i++){
        for(j = 0; j < a->topk; j++){
            if(mloss[i][j] > 0) avf[i][j] = (mloss[i][j] - min) / factor;
        }
    }
}
