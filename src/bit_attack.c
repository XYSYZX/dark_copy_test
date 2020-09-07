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
    attack_args *a = net->attack;
    int layer_num = net->n;
    int topk = a->topk;
    int i;
    if(a->a_input){
        a->layer_num = 1;
        //a->topk_inputs = topk;

        a->mloss_loc_inputs = make_2d_array_int(1, topk * a->total_img / a->n);
        a->mloss_inputs = make_2d_array_float(1, topk * a->total_img / a->n);
        a->mloss_loc = a->mloss_loc_inputs;
        a->mloss = a->mloss_inputs;

        a->inputs_len = (int *)calloc(1, sizeof(int));
        a->inputs_len[0] = net->inputs*net->batch;
        a->len = a->inputs_len;

        a->input_grads = (float **)calloc(1, sizeof(float*));
        a->input_grads_gpu = (float **)calloc(1, sizeof(float*));
        a->input_grads[0] = net->delta;
        a->input_grads_gpu[0] = net->delta_gpu;
        a->grads = a->input_grads;
        a->grads_gpu = a->input_grads_gpu;

        a->inputs = (float **)calloc(1, sizeof(float*));
        a->inputs_gpu = (float **)calloc(1, sizeof(float*));
        a->inputs[0] = net->input;
        a->inputs_gpu[0] = net->input_gpu;
        a->x = a->inputs;
        a->x_gpu = a->inputs_gpu;
    }
    if(a->a_weight){
        a->layer_num = net->n;
        //a->topk_weights = topk;

        a->mloss_loc_weights = make_2d_array_int(layer_num, topk * a->total_img / a->n);
        a->mloss_weights = make_2d_array_float(layer_num, topk * a->total_img / a->n);
        a->mloss_loc = a->mloss_loc_weights;
        a->mloss = a->mloss_weights;

        a->weights_len = (int *)calloc(net->n, sizeof(int));
        for(i = 0; i < net->n; i++) a->weights_len[i] = net->layers[i].nweights;
        a->len = a->weights_len;

        a->weight_grads = (float **)calloc(net->n, sizeof(float*));
        a->weight_grads_gpu = (float **)calloc(net->n, sizeof(float*));
        for(i = 0; i < net->n; i++) a->weight_grads[i] = net->layers[i].weight_updates;
        for(i = 0; i < net->n; i++) a->weight_grads_gpu[i] = net->layers[i].weight_updates_gpu;
        a->grads = a->weight_grads;
        a->grads_gpu = a->weight_grads_gpu;

        a->weights = (float **)calloc(net->n, sizeof(float*));
        a->weights_gpu = (float **)calloc(net->n, sizeof(float*));
        for(i = 0; i < net->n; i++) a->weights[i] = net->layers[i].weights;
        for(i = 0; i < net->n; i++) a->weights_gpu[i] = net->layers[i].weights_gpu;
        a->x = a->weights;
        a->x_gpu = a->weights_gpu;
    }
    if(a->a_bias){
        a->layer_num = net->n;
        //a->topk_biases = topk;

        a->mloss_loc_biases = make_2d_array_int(layer_num, topk * a->total_img / a->n);
        a->mloss_biases = make_2d_array_float(layer_num, topk * a->total_img / a->n);
        a->mloss_loc = a->mloss_loc_biases;
        a->mloss = a->mloss_biases;

        a->biases_len = (int *)calloc(net->n, sizeof(int));
        for(int i = 0; i < net->n; i++) a->biases_len[i] = net->layers[i].n;
        a->len = a->biases_len;

        a->bias_grads = (float **)calloc(net->n, sizeof(float*));
        a->bias_grads_gpu = (float **)calloc(net->n, sizeof(float*));
        for(i = 0; i < net->n; i++) a->bias_grads[i] = net->layers[i].bias_updates;
        for(i = 0; i < net->n; i++) a->bias_grads_gpu[i] = net->layers[i].bias_updates_gpu;
        a->grads = a->bias_grads;
        a->grads_gpu = a->bias_grads_gpu;

        a->biases = (float **)calloc(net->n, sizeof(float*));
        a->biases_gpu = (float **)calloc(net->n, sizeof(float*));
        for(i = 0; i < net->n; i++) a->biases[i] = net->layers[i].biases;
        for(i = 0; i < net->n; i++) a->biases_gpu[i] = net->layers[i].biases_gpu;
        a->x = a->biases;
        a->x_gpu = a->biases_gpu;
    }
    if(a->a_output){
        a->layer_num = net->n;
        //a->topk_outputs = topk;

        a->mloss_loc_outputs = make_2d_array_int(layer_num, topk * a->total_img / a->n);
        a->mloss_outputs = make_2d_array_float(layer_num, topk * a->total_img / a->n);
        a->mloss_loc = a->mloss_loc_outputs;
        a->mloss = a->mloss_outputs;

        a->outputs_len = (int *)calloc(net->n, sizeof(int));
        for(int i = 0; i < net->n; i++) a->outputs_len[i] = net->layers[i].outputs*net->batch;
        a->len = a->outputs_len;

        a->output_grads = (float **)calloc(net->n, sizeof(float*));
        a->output_grads_gpu = (float **)calloc(net->n, sizeof(float*));
        for(i = 0; i < net->n; i++) a->output_grads[i] = net->layers[i].delta;
        for(i = 0; i < net->n; i++) a->output_grads_gpu[i] = net->layers[i].delta_gpu;
        a->grads = a->output_grads;
        a->grads_gpu = a->output_grads_gpu;

        a->outputs = (float **)calloc(net->n, sizeof(float*));
        a->outputs_gpu = (float **)calloc(net->n, sizeof(float*));
        for(i = 0; i < net->n; i++) a->outputs[i] = net->layers[i].output;
        for(i = 0; i < net->n; i++) a->outputs_gpu[i] = net->layers[i].output_gpu;
        a->x = a->outputs;
        a->x_gpu = a->outputs_gpu;
    }
}

void free_attack_args(attack_args a)
{
    int layer_num = a.layer_num;
    {
        if(a.mloss_loc_inputs) free_2d_array_int(a.mloss_loc_inputs, 1);
        if(a.mloss_inputs) free_2d_array_float(a.mloss_inputs, 1);
        if(a.inputs_len) free(a.inputs_len);
        if(a.input_grads) free(a.input_grads);
        if(a.input_grads_gpu) free(a.input_grads_gpu);
        if(a.inputs) free(a.inputs);
        if(a.inputs_gpu) free(a.inputs_gpu);
    }
    {
        if(a.mloss_loc_weights) free_2d_array_int(a.mloss_loc_weights, layer_num);
        if(a.mloss_weights) free_2d_array_float(a.mloss_weights, layer_num);
        if(a.weights_len) free(a.weights_len);
        if(a.weight_grads) free(a.weight_grads);
        if(a.weight_grads_gpu) free(a.weight_grads_gpu);
        if(a.weights) free(a.weights);
        if(a.weights_gpu) free(a.weights_gpu);
    }
    {
        if(a.mloss_loc_biases) free_2d_array_int(a.mloss_loc_biases, layer_num);
        if(a.mloss_biases) free_2d_array_float(a.mloss_biases, layer_num);
        if(a.biases_len) free(a.biases_len);
        if(a.bias_grads) free(a.bias_grads);
        if(a.bias_grads_gpu) free(a.bias_grads_gpu);
        if(a.biases) free(a.biases);
        if(a.biases_gpu) free(a.biases_gpu);
    }
    {
        if(a.mloss_loc_outputs) free_2d_array_int(a.mloss_loc_outputs, layer_num);
        if(a.mloss_outputs) free_2d_array_float(a.mloss_outputs, layer_num);
        if(a.outputs_len) free(a.outputs_len);
        if(a.output_grads) free(a.output_grads);
        if(a.output_grads_gpu) free(a.output_grads_gpu);
        if(a.outputs) free(a.outputs);
        if(a.outputs_gpu) free(a.outputs_gpu);
    }
}

void attack_data(network *net, load_args args)
{
    float avg_maxloss = 0;
    float avg_loss = 0;
    float loss = 0;
    float maxloss = 0;
    int i, j, k;
    double time;

    attack_args *attack = net->attack;
    int iter = attack->total_img / attack->n;
    set_attack_args(net);

    
    data val, buffer;
    args.d = &buffer;

    pthread_t load_thread = load_data(args);
    attack->seen_img += attack->n;

    for(i = 0; i < iter; i++){
        attack->iter_idx = i;
        pthread_join(load_thread, 0);
        val = buffer;
        printf("images: %d\n", attack->seen_img);

        loss = network_predict_search(net, val);
        time = what_time_is_it_now();
        for(j = 0; j < attack->layer_num; j++){
            attack->layer_idx = j;
            //printf("grad: %x, gpu: %x\n", attack->grads[j], attack->grads_gpu[j]);
            if(attack->grads[j] && attack->grads_gpu[j]){
                get_topk_grad(attack->grads[j], attack->grads_gpu[j], attack->len[j], attack->mloss_loc[j] + i*attack->topk, attack->topk);
                //for(int z = 0; z < attack->topk; z++) printf("%f ", attack->grads[j][*(attack->mloss_loc[j]+i*attack->topk+z)]);
                if(attack->progress_attack){
                    if(attack->a_weight || attack->a_bias){
                        progressive_attack(net);
                    }
                    attack->mloss[j][attack->topk*i] = network_predict_attack(net, val);
                    if(attack->a_weight || attack->a_bias){
                        progressive_attack(net);
                    }
                    printf("progress: %f\n", attack->mloss[j][attack->topk*i]);
                    maxloss = maxloss > attack->mloss[j][attack->topk*i]? maxloss: attack->mloss[j][attack->topk*i];
                }
                else{
                    for(k = 0; k < attack->topk; k++){
                        attack->k_idx = k;
                        if(attack->a_weight || attack->a_bias){
                            single_attack(net);
                        }
                        attack->mloss[j][i*attack->topk + k] = network_predict_attack(net, val);
                        if(attack->a_weight || attack->a_bias){
                            single_attack(net);
                        }
                        printf("single: %f\n", attack->mloss[j][i*attack->topk + k]);
                        maxloss = maxloss > attack->mloss[j][i*attack->topk + k]? maxloss: attack->mloss[j][i*attack->topk + k];
                    }
                }
            }
        }
        printf("in iter %d, time use: %f\n", i, what_time_is_it_now() - time);
        printf("iter: %d, orig loss: %f, attacked loss: %f\n", i, loss, maxloss);
        avg_loss += loss;
        avg_maxloss += maxloss;
        loss = 0;
        maxloss = 0;
        load_thread = load_data(args);
        attack->seen_img += attack->n;
    }
    avg_loss /= iter;
    avg_maxloss /= iter;
    printf("average orig loss: %f, attacked loss: %f\n", avg_loss, avg_maxloss);
    if(1){
        printf("location of max losses: \n");
        print_2d_array_int(attack->mloss_loc, attack->layer_num, attack->topk * iter);
    }
    int **maxlosses = make_2d_array_int(attack->layer_num, attack->topk);
    int **maxlosses_loc = make_2d_array_int(attack->layer_num, attack->topk);
    get_max(attack, maxlosses, maxlosses_loc);
    printf("frequency final max losses: \n");
    print_2d_array_int(maxlosses, attack->layer_num, attack->topk);
    printf("final max losses location: \n");
    print_2d_array_int(maxlosses_loc, attack->layer_num, attack->topk);
    free_2d_array_int(maxlosses_loc, attack->layer_num);
    free_2d_array_int(maxlosses, attack->layer_num);
    //printf("have a look\n");
    //for(int z = 0; z < attack->topk * attack->total_img / attack->n; z++) printf("%f ", attack->grads[0][*(attack->mloss_loc[0]+z)]);
    free_attack_args(*attack);
}

void progressive_attack(network *net)
{
    attack_args a = *(net->attack);
    if(a.sign_attack){
        int offset = a.iter_idx * a.topk;
        if(a.a_input){
            cuda_pull_array(a.grads_gpu[a.layer_idx], a.grads[a.layer_idx], a.len[a.layer_idx]);
            if(a.reverse == 1) sign_attacker(a.x[a.layer_idx], a.mloss_loc[a.layer_idx]+offset, a.topk, a.grads[a.layer_idx], a.epsilon);
            else sign_delete(a.x[a.layer_idx], a.mloss_loc[a.layer_idx]+offset, a.topk, a.grads[a.layer_idx], a.epsilon);
        }
        else{
            if(a.reverse == 1){
            #ifdef GPU
                sign_attacker_gpu(a.x_gpu[a.layer_idx], a.mloss_loc[a.layer_idx]+offset, a.topk, a.grads_gpu[a.layer_idx], a.epsilon);
            #else
                sign_attacker(a.x[a.layer_idx], a.mloss_loc[a.layer_idx]+offset, a.topk, a.grads[a.layer_idx], a.epsilon);
            #endif
            }
            else{
            #ifdef GPU
                sign_delete_gpu(a.x_gpu[a.layer_idx], a.mloss_loc[a.layer_idx]+offset, a.topk, a.grads_gpu[a.layer_idx], a.epsilon);
            #else
                sign_delete(a.x[a.layer_idx], a.mloss_loc[a.layer_idx]+offset, a.topk, a.grads[a.layer_idx], a.epsilon);
            #endif
            }
        }
    }
    else{
        if(a.a_input){
            for(int i = 0; i < a.topk; i++){
                int offset = a.iter_idx * a.topk + i;
                for(int j = 0; j < a.fb_len; j++){
                    inject_noise_float_onebit(a.x[a.layer_idx], a.mloss_loc[a.layer_idx][offset], a.flipped_bit[j]);
                }
            }
        }
        else{
            for(int i = 0; i < a.topk; i++){
                int offset = a.iter_idx * a.topk + i;
                for(int j = 0; j < a.fb_len; j++){
                #ifdef GPU
                    inject_noise_float_onebit_gpu(a.x_gpu[a.layer_idx], a.mloss_loc[a.layer_idx][offset], a.flipped_bit[j]);
                #else
                    inject_noise_float_onebit(a.x[a.layer_idx], a.mloss_loc[a.layer_idx][offset], a.flipped_bit[j]);
                #endif
                }
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
    if(a.a_input){
        int offset = a.iter_idx * a.topk + a.k_idx;
        for(int i = 0; i < a.fb_len; i++){
            inject_noise_float_onebit(a.x[a.layer_idx], a.mloss_loc[a.layer_idx][offset], a.flipped_bit[i]);
        }
    }
    else{
        int offset = a.iter_idx * a.topk + a.k_idx;
        for(int i = 0; i < a.fb_len; i++){
            #ifdef GPU
            inject_noise_float_onebit_gpu(a.x_gpu[a.layer_idx], a.mloss_loc[a.layer_idx][offset], a.flipped_bit[i]);
            #else
            inject_noise_float_onebit(a.x[a.layer_idx], a.mloss_loc[a.layer_idx][offset], a.flipped_bit[i]);
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
    printf("attack sign!\n");
    for(i = 0; i < topk; i++){
        idx = loc[i];
        float x_sign = sign(grad[idx]);
        x[idx] += epsilon * x_sign;
    }
}

void sign_delete(float *x, int *loc, int topk, float *grad, float epsilon)
{
    int i, idx;
    printf("delete sign!\n");
    for(i = 0; i < topk; i++){
        idx = loc[i];
        float x_sign = sign(grad[idx]);
        x[idx] -= epsilon * x_sign;
    }
}

void get_max(attack_args *attack, int **x, int **x_loc)
{
    int i, j;
    int *lens = attack->len;
    int layer_num = attack->layer_num;
    int topk = attack->topk;
    int iter = attack->total_img / attack->n;
    int t = iter * topk;
    int **mloss_loc = attack->mloss_loc;

    for(i = 0; i < layer_num; i++){
        int *cal = (int *)calloc(lens[i], sizeof(int));
        for(j = 0; j < t; j++){
            cal[mloss_loc[i][j]] += 1;
        }
        top_k_int(cal, lens[i], topk, x_loc[i], x[i]);
        free(cal);
    }
}

