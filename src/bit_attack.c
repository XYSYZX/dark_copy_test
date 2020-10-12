#include <float.h>
#include "bit_attack.h"

extern int detections_comparator(const void *pa, const void *pb);

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
    a->layer_num = layer_num;
    int iter = a->iter;
    int fb_len = a->fb_len;

    
    //input
    /*
    topk = a->topk_inputs;
    a->grads_loc_inputs = make_2d_array_int(1, topk * iter);
    a->mloss_loc_inputs = make_2d_array_int(1, topk);
    a->mloss_inputs = make_2d_array_float(1, topk * fb_len);
    a->macc_inputs = make_2d_array_float(1, topk * fb_len);
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
    */
    //weight
    topk = a->topk_weight;
    a->grads_loc_weights = make_2d_array_int(layer_num, topk * iter);
    a->grads_val_weights = make_2d_array_float(layer_num, topk * iter);
    a->mloss_loc_weights = make_2d_array_int(2, topk);
    a->mloss_weights = (float *)calloc(topk * fb_len, sizeof(float));
    a->macc_weights = (float *)calloc(topk * fb_len, sizeof(float));
    a->avf_weights = (float *)calloc(topk, sizeof(float));

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

    a->topk_weights = (int *)calloc(net->n, sizeof(int));
    for(i = 0; i < net->n; i++){
        if(a->weights_len[i] > a->topk_weight) a->topk_weights[i] = a->topk_weight;
        else a->topk_weights[i] = a->weights_len[i];
    }

    //bias
    topk = a->topk_bias;
    a->grads_loc_biases = make_2d_array_int(layer_num, topk * iter);
    a->grads_val_biases = make_2d_array_float(layer_num, topk * iter);
    a->mloss_loc_biases = make_2d_array_int(2, topk);
    a->mloss_biases = (float *)calloc(topk * fb_len, sizeof(float));
    a->macc_biases = (float *)calloc(topk * fb_len, sizeof(float));
    a->avf_biases = (float *)calloc(topk, sizeof(float));

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
    
    a->topk_biases = (int *)calloc(net->n, sizeof(int));
    for(i = 0; i < net->n; i++){
        if(a->biases_len[i] > a->topk_bias) a->topk_biases[i] = a->topk_bias;
        else a->topk_biases[i] = a->biases_len[i];
    }
    //output
    /*
    topk = a->topk_outputs;
    a->grads_loc_outputs = make_2d_array_int(layer_num, topk * iter);
    a->mloss_loc_outputs = make_2d_array_int(layer_num, topk);
    a->mloss_outputs = make_2d_array_float(layer_num, topk * fb_len);
    a->macc_outputs = make_2d_array_float(layer_num, topk * fb_len);
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
    */
}

void free_attack_args(attack_args a)
{
    int layer_num = a.layer_num;
    /*
    {
        if(a.grads_loc_inputs) free_2d_array_int(a.grads_loc_inputs, 1);
        if(a.mloss_loc_inputs) free_2d_array_int(a.mloss_loc_inputs, 1);
        if(a.mloss_inputs) free_2d_array_float(a.mloss_inputs, 1);
        if(a.macc_inputs) free_2d_array_float(a.macc_inputs, 1);
        if(a.avf_inputs) free_2d_array_float(a.avf_inputs, 1);
        if(a.inputs_len) free(a.inputs_len);
        if(a.input_grads) free(a.input_grads);
        if(a.input_grads_gpu) free(a.input_grads_gpu);
        if(a.inputs) free(a.inputs);
        if(a.inputs_gpu) free(a.inputs_gpu);
    }*/
    {
        if(a.grads_loc_weights) free_2d_array_int(a.grads_loc_weights, layer_num);
        if(a.grads_val_weights) free_2d_array_float(a.grads_val_weights, layer_num);
        if(a.mloss_loc_weights) free_2d_array_int(a.mloss_loc_weights, 2);
        if(a.mloss_weights) free(a.mloss_weights);
        if(a.macc_weights) free(a.macc_weights);
        if(a.avf_weights) free(a.avf_weights);
        if(a.weights_len) free(a.weights_len);
        if(a.weight_grads) free(a.weight_grads);
        if(a.weight_grads_gpu) free(a.weight_grads_gpu);
        if(a.weights) free(a.weights);
        if(a.weights_gpu) free(a.weights_gpu);
    }
    {
        if(a.grads_loc_biases) free_2d_array_int(a.grads_loc_biases, layer_num);
        if(a.grads_val_biases) free_2d_array_float(a.grads_val_biases, layer_num);
        if(a.mloss_loc_biases) free_2d_array_int(a.mloss_loc_biases, 2);
        if(a.mloss_biases) free(a.mloss_biases);
        if(a.macc_biases) free(a.macc_biases);
        if(a.avf_biases) free(a.avf_biases);
        if(a.biases_len) free(a.biases_len);
        if(a.bias_grads) free(a.bias_grads);
        if(a.bias_grads_gpu) free(a.bias_grads_gpu);
        if(a.biases) free(a.biases);
        if(a.biases_gpu) free(a.biases_gpu);
    }
    /*
    {
        if(a.grads_loc_outputs) free_2d_array_int(a.grads_loc_outputs, layer_num);
        if(a.mloss_loc_outputs) free_2d_array_int(a.mloss_loc_outputs, layer_num);
        if(a.mloss_outputs) free_2d_array_float(a.mloss_outputs, layer_num);
        if(a.macc_outputs) free_2d_array_float(a.macc_outputs, layer_num);
        if(a.avf_outputs) free_2d_array_float(a.avf_outputs, layer_num);
        if(a.outputs_len) free(a.outputs_len);
        if(a.output_grads) free(a.output_grads);
        if(a.output_grads_gpu) free(a.output_grads_gpu);
        if(a.outputs) free(a.outputs);
        if(a.outputs_gpu) free(a.outputs_gpu);
    }*/
}

void attack_data(network *net, load_args args, load_args val_args)
{
    double loss = 0;
    int i, j;
    int type;

    attack_args *attack = net->attack;
    int iter = args.m / args.n;
    attack->iter = iter;
    set_attack_args(net);
    set_batch_network(net, args.n);

    data val, buffer;
    args.d = &buffer;

    pthread_t load_thread = load_data_in_thread(args);
    attack->seen_img += args.n;
    args.paths += args.n;

    for(i = 0; i < iter; i++){
        double time = what_time_is_it_now();
        attack->iter_idx = i;
        pthread_join(load_thread, 0);
        val = buffer;
        fprintf(stderr, "images: %d\n", attack->seen_img);
        loss += network_predict_search(net, val) / iter;
        fprintf(stderr, "detect time: %f\n", what_time_is_it_now() - time);
        for(j = 0; j < attack->layer_num; j++){
            //printf("grad: %x, gpu: %x\n", attack->grads[j], attack->grads_gpu[j]);
            attack->layer_idx = j;
            get_topk_grad(attack);
        }
        load_thread = load_data_in_thread(args);
        free_data(val);
        attack->seen_img += args.n;
        args.paths += args.n;
        fprintf(stderr, "get topk time: %f\n", what_time_is_it_now() - time);
    }
    get_max_loss(attack);

    attack->seen_img = 0;
    FILE *avf_fp = fopen(attack->avg_log, "w+");
    if(!avf_fp){
        printf("no avg file!");
        return;
    }
    for(i = 0; i < 2; i++){
        type = i;
        get_avf(net, val_args, type, avf_fp);
    }

    free_attack_args(*attack);
}

void get_topk_grad(attack_args *a)
{
    int j = a->layer_idx;
    int i = a->iter_idx;
    /*
    if(j == 0){
        if(a->input_grads[j] && a->input_grads_gpu[j]){
            get_topk(a->input_grads[j], a->input_grads_gpu[j], a->inputs_len[j], a->grads_loc_inputs[j]+i*a->topk_inputs, a->topk_inputs);
        }
    }*/
    if(a->weight_grads[j] && a->weight_grads_gpu[j]){
        get_topk(a->weight_grads[j], a->weight_grads_gpu[j], a->weights_len[j], a->grads_loc_weights[j]+i*a->topk_weight, a->grads_val_weights[j]+i*a->topk_weight, a->topk_weights[j]);
    }
    if(a->bias_grads[j] && a->bias_grads_gpu[j]){
        get_topk(a->bias_grads[j], a->bias_grads_gpu[j], a->biases_len[j], a->grads_loc_biases[j]+i*a->topk_bias, a->grads_val_biases[j]+i*a->topk_bias, a->topk_biases[j]);
    }
    /*
    if(a->output_grads[j] && a->output_grads_gpu[j]){
        get_topk(a->output_grads[j], a->output_grads_gpu[j], a->outputs_len[j], a->grads_loc_outputs[j]+i*a->topk_outputs, a->topk_outputs);
    }*/
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
            bit_flip_attacker(a);
        }
        else{
        #ifdef GPU
            bit_flip_attacker_gpu(a);
        #else
            bit_flip_attacker(a);
        #endif
        }
    }
    if(a.a_weight || a.a_bias) {
        net->attack->reverse *= -1;
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

void bit_flip_attacker(attack_args a)
{
    int layer_idx = a.mloss_loc[0][a.k_idx];
    int idx = a.mloss_loc[1][a.k_idx];
    float *x = a.x[layer_idx];
    int bit_idx = a.bit_idx;
    inject_noise_float_onebit(x, idx, bit_idx);
}

void get_max_loss(attack_args *a)
{
    int i, j;
    int layer_num = a->layer_num;
    int iter = a->iter;
    //int *mloss_freq_weights = (int *)calloc(layer_num, a->topk_weights, sizeof(int));
    //int *mloss_freq_biases = (int *)calloc(layer_num, a->topk_biases, sizeof(int));
    float *mloss_val_weights = (float *)calloc(layer_num * a->topk_weight, sizeof(float));
    float *mloss_val_biases = (float *)calloc(layer_num * a->topk_bias, sizeof(float));
    int *mloss_loc_weights = (int *)calloc(layer_num * a->topk_weight, sizeof(int));
    int *mloss_loc_biases = (int *)calloc(layer_num * a->topk_bias, sizeof(int));

    int count = 0;

    for(i = 0; i < layer_num; i++){
        if(a->weights_len[i]){
            count = i * a->topk_weight;
            if(1){
                //printf("length of weights_len[i]: %d\n", a->weights_len[i]);
                int *count_weight = (int *)calloc(a->weights_len[i], sizeof(int));
                float *val_weight = (float *)calloc(a->weights_len[i], sizeof(float));
                for(j = 0; j < iter * a->topk_weight ; j++){
                    if(a->grads_loc_weights[i][j] > 0){
                        count_weight[a->grads_loc_weights[i][j]] += 1;
                        val_weight[a->grads_loc_weights[i][j]] += a->grads_val_weights[i][j];
                    }
                }
                for(j = 0; j < iter * a->topk_weight; j++){
                    if(a->grads_loc_weights[i][j] > 0){
                        val_weight[a->grads_loc_weights[i][j]] /= count_weight[a->grads_loc_weights[i][j]];
                    }
                }
                get_topk_int(count_weight, val_weight, a->weights_len[i], a->topk_weights[i], mloss_loc_weights+count, mloss_val_weights+count);
                free(count_weight);
                free(val_weight);
            }
            if(0){
                top_k_with_idx(a->grads_val_weights[i], a->grads_loc_weights[i], iter*a->topk_weight, a->topk_weights[i], mloss_val_weights + count, mloss_loc_weights + count);
            }
        }
        if(a->biases_len[i]){
            count = i * a->topk_bias;
            if(1){
                //printf("length of biases_len[i]: %d\n", a->biases_len[i]);
                int *count_bias = (int *)calloc(a->biases_len[i], sizeof(int));
                float *val_bias = (float *)calloc(a->biases_len[i], sizeof(float));
                for(j = 0; j < iter * a->topk_bias ; j++){
                    //printf("count_bias idx: %d\n", a->grads_loc_biases[i][j]);
                    if(a->grads_loc_biases[i][j] > 0){
                        count_bias[a->grads_loc_biases[i][j]] += 1;
                        val_bias[a->grads_loc_biases[i][j]] += a->grads_val_biases[i][j];
                    }
                }
                for(j = 0; j < iter * a->topk_bias ; j++){
                    if(a->grads_loc_biases[i][j] > 0){
                        val_bias[a->grads_loc_biases[i][j]] /= count_bias[a->grads_loc_biases[i][j]];
                    }
                }
                get_topk_int(count_bias, val_bias, a->biases_len[i], a->topk_biases[i], mloss_loc_biases+count, mloss_val_biases+count);
                free(count_bias);
                free(val_bias);
            }
            if(0){
                top_k_with_idx(a->grads_val_biases[i], a->grads_loc_biases[i], iter*a->topk_bias, a->topk_biases[i], mloss_val_biases + count, mloss_loc_biases + count);
            }
        }
    }
    
    printf("value and loc of bias\n");
    for(int i = 0; i < layer_num; i++){
        for(int j = 0; j < a->topk_bias; j++){
            printf("%f ", mloss_val_biases[i*a->topk_bias+j]);
        }
        printf("\n");
    }
    for(int i = 0; i < layer_num; i++){
        for(int j = 0; j < a->topk_bias; j++){
            printf("%d ", mloss_loc_biases[i*a->topk_bias+j]);
        }
        printf("\n");
    }
    
    get_topk_with_layer(mloss_val_weights, mloss_loc_weights, a->topk_weight*layer_num, a->topk_weight, a->mloss_loc_weights[0], a->mloss_loc_weights[1]);
    get_topk_with_layer(mloss_val_biases, mloss_loc_biases, a->topk_bias*layer_num, a->topk_bias, a->mloss_loc_biases[0], a->mloss_loc_biases[1]);
    qsort_with_layer(mloss_val_weights, a->mloss_loc_weights[0], a->mloss_loc_weights[1], 0, a->topk_weight-1);
    qsort_with_layer(mloss_val_biases, a->mloss_loc_biases[0], a->mloss_loc_biases[1], 0, a->topk_bias-1);

    printf("max layer loc of bias\n");
    for(i = 0; i < a->topk_bias; i++) printf("%f ", mloss_val_biases[i]);
    printf("\n");
    for(i = 0; i < a->topk_bias; i++) printf("%d ", a->mloss_loc_biases[0][i]);
    printf("\n");
    for(i = 0; i < a->topk_bias; i++) printf("%d ", a->mloss_loc_biases[1][i]);
    printf("\n");
    
    free(mloss_val_weights);
    free(mloss_val_biases);
    free(mloss_loc_weights);
    free(mloss_loc_biases);
    //free_2d_array_int(mloss_freq_output, layer_num);
}

void get_avf(network *net, load_args args, int type, FILE *avf_fp)
{
    attack_args *a = net->attack;
    float avg_loss = 0;
    float avg_acc = 0;
    int layer_num = net->n;

    int i, k, m;

    //if(a->sign_attack) args.n = net->batch;
    //int iter = args.m / args.n;

    data val, buffer;
    args.d = &buffer;

    set_batch_network(net, args.n);

    int iter = args.m / args.n;
    a->iter = iter;

    const float nms = 0.5;
    const float thresh = 0.2;
    const float iou_thresh = 0.2;
    const float thresh_calc_avg_iou = 0.01;
    int classes = net->layers[net->n-1].classes;

    switch(type){
        case 0:
            a->a_weight = 1;
            a->layer_num = layer_num;
            a->grads = a->weight_grads;
            a->grads_gpu = a->weight_grads_gpu;
            a->len = a->weights_len;
            a->x = a->weights;
            a->x_gpu = a->weights_gpu;
            a->topk = a->topk_weight;
            a->mloss = a->mloss_weights;
            a->macc = a->macc_weights;
            a->mloss_loc = a->mloss_loc_weights;
            a->avf = a->avf_weights;
            break;
        case 1:
            a->a_bias = 1;
            a->layer_num = layer_num;
            a->grads = a->bias_grads;
            a->grads_gpu = a->bias_grads_gpu;
            a->len = a->biases_len;
            a->x = a->biases;
            a->x_gpu = a->biases_gpu;
            a->topk = a->topk_bias;
            a->mloss = a->mloss_biases;
            a->macc = a->macc_biases;
            a->mloss_loc = a->mloss_loc_biases;
            a->avf = a->avf_biases;
            break;
        default:
            printf("wrong attack type!\n");
            return;
    }

    args.path = args.paths[0];
    char labelpath[4096];
    int count_boxes = 0;
    find_replace(args.path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    args.boxes = read_boxes(labelpath, &count_boxes);
    args.num_boxes = count_boxes;

    pthread_t load_thread = load_data_in_thread(args);
    a->seen_img += args.n;

    for(i = 0; i < iter; i++){
        pthread_join(load_thread, 0);
        val = buffer;

        a->iter_idx = i;
        fprintf(stderr, "images: %d\n", a->seen_img);

        avg_loss += network_predict_search(net, val) / iter;
        int nboxes = 0;
        detection *dets = get_network_boxes(net, net->w, net->h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, classes, nms);

        avg_acc += cal_map(net, dets, args.boxes, nboxes, args.num_boxes, iou_thresh, thresh_calc_avg_iou) / iter;
        free_detections(dets, nboxes);
        //printf("avg_loss: %f, avg_acc: %f\n", avg_loss, avg_acc);

        for(k = 0; k < a->topk; k++){
            a->k_idx = k;
            for(m = 0; m < a->fb_len; m++){
                //double time = what_time_is_it_now();
                a->bit_idx = a->flipped_bit[m];
                if(a->a_weight || a->a_bias){
                    single_attack(net);
                }

                a->mloss[a->fb_len*k+m] += network_predict_attack(net, val) / iter;
                int nboxes = 0;
                detection *dets = get_network_boxes(net, net->w, net->h, thresh, .5, 0, 1, &nboxes); 
                if (nms) do_nms_sort(dets, nboxes, classes, nms);

                a->macc[a->fb_len*k+m] += cal_map(net, dets, args.boxes, nboxes, args.num_boxes, iou_thresh, thresh_calc_avg_iou) / iter;
                free_detections(dets, nboxes);

                if(a->a_weight || a->a_bias){
                    single_attack(net);
                }
                //fprintf(stderr, "attack time: %f\n", what_time_is_it_now() - time);
            }
        }
        free(args.boxes);

        args.path = args.paths[i+1];
        char labelpath[4096];
        int count_boxes = 0;
        find_replace(args.path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);
        args.boxes = read_boxes(labelpath, &count_boxes);
        args.num_boxes = count_boxes;

        load_thread = load_data_in_thread(args);
        free_data(val);
        a->seen_img += args.n;
    }

    a->loss_thresh = avg_loss;
    a->acc_thresh = avg_acc;
    printf("type: %d\n", type);
    fprintf(avf_fp, "type: %d\n", type);
    cal_avf(a, avf_fp);
    {
        for(i = 0; i < a->topk; i++) printf("%f ", a->avf[i]);
        printf("\n");
    }
    print_avf_log(a, avf_fp);
    //if(type == 0) a->a_input = 0;
    if(type == 0) a->a_weight = 0;
    if(type == 1) a->a_bias = 0;
    //if(type == 3) a->a_output = 0;
    a->seen_img = 0;
}

void cal_avf(attack_args *a, FILE *fp)
{
    int i, j, m;
    int topk = a->topk;
    int fb_len = a->fb_len;
    float *max_loss = (float *)calloc(fb_len, sizeof(float));
    float *min_loss = (float *)calloc(fb_len, sizeof(float));
    float *losses = (float *)calloc(topk, sizeof(float));
    float *max_acc = (float *)calloc(fb_len, sizeof(float));
    float *min_acc = (float *)calloc(fb_len, sizeof(float));
    float *accs = (float *)calloc(topk, sizeof(float));
    for(i = 0; i < fb_len; i++){
        min_loss[i] = FLT_MAX;
        min_acc[i] = FLT_MAX;
    }

    float *avf = a->avf;
    float *mloss = a->mloss;
    float *macc = a->macc;

    float t_loss;
    float t_acc;
    //float loss_thresh = a->loss_thresh;
    //float acc_thresh = a->acc_thresh;
    printf("mloss and macc: \n");
    for(i = 0; i < topk*fb_len; i++) printf("%f ", mloss[i]);
    printf("\n");
    for(i = 0; i < topk*fb_len; i++) printf("%f ", macc[i]);
    printf("\n");

    for(j = 0; j < topk; j++){
        for(m = 0; m < fb_len; m++){
            t_loss = mloss[j*fb_len+m];
            t_acc = macc[j*fb_len+m];
            if(isnan(t_loss)|| isinf(t_loss) || max_loss[m] < t_loss)
                max_loss[m] = t_loss;
            if(isnan(t_loss)|| isinf(t_loss) || min_loss[m] > t_loss)
                min_loss[m] = t_loss;
            if(max_acc[m] < t_acc)
                max_acc[m] = t_acc;
            if(min_acc[m] > t_acc)
                min_acc[m] = t_acc;
        }
    }

    for(j = 0; j < topk; j++){
        for(m = 0; m < fb_len; m++){
            t_loss = mloss[j*fb_len+m];
            t_acc = macc[j*fb_len+m];
            if(isnan(t_loss) || isinf(t_loss) || isnan(min_loss[m]) || isinf(min_loss[m]) || isnan(max_loss[m]) || isinf(max_loss[m])) mloss[j*fb_len+m] = 1;
            else if(min_loss[m] == max_loss[m]) mloss[j*fb_len+m] = 0.5;
            else mloss[j*fb_len+m] = (mloss[j*fb_len+m] - min_loss[m]) / (max_loss[m] - min_loss[m]); 
            //if(isnan(t_acc) || isinf(t_acc) || isnan(min_acc[m]) || isinf(min_acc[m]) || isnan(max_acc[m]) || isinf(max_acc[m])) macc[i][j*fb_len+m] = 1;
            if(max_acc[m] == 0) macc[j*fb_len+m] = 0;
            else if(min_acc[m] == max_acc[m]) macc[j*fb_len+m] = 0.5;
            else macc[j*fb_len+m] = (macc[j*fb_len+m] - min_acc[m]) / (max_acc[m] - min_acc[m]); 
            losses[j] += mloss[j*fb_len+m];
            accs[j] += macc[j*fb_len+m];
        }
        losses[j] = losses[j] / fb_len;
        accs[j] = accs[j] / fb_len;
    }

    for(j = 0; j < topk; j++){
        avf[j] = losses[j] * a->alpha + (1 - accs[j]) * (1 - a->alpha) ;
    }
    printf("losses and accs: \n");
    for(i = 0; i < topk; i++) printf("%f ", losses[i]);
    printf("\n");
    for(i = 0; i < topk; i++) printf("%f ", accs[i]);
    printf("\n");

    fprintf(fp, "losses and accs: \n");
    for(i = 0; i < topk; i++) fprintf(fp, "%f ", losses[i]);
    fprintf(fp, "\n");
    for(i = 0; i < topk; i++) fprintf(fp, "%f ", accs[i]);
    fprintf(fp, "\n");

    free(losses);
    free(accs);
    free(max_loss);
    free(min_loss);
    free(max_acc);
    free(min_acc);
}
/*
int detections_comparator(const void *pa, const void *pb)
{
    box_prob a = *(const box_prob *)pa;
    box_prob b = *(const box_prob *)pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}
*/
float cal_map(network *net, detection *dets, box_label *truth, int nboxes, int num_labels, float iou_thresh, float thresh_calc_avg_iou)
{
    int i, j;
    int classes = net->layers[net->n-1].classes;

    box_prob* detections = calloc(1, sizeof(box_prob));
    int detections_count = 0;

    int *truth_classes_count = calloc(classes, sizeof(int));

    for (i = 0; i < num_labels; ++i) {
        truth_classes_count[truth[i].id]++;   //每张图片中真实存在的物体（txt文件）
    }
    for(i = 0; i < nboxes; i++){
        int class_id;
        for (class_id = 0; class_id < classes; ++class_id) {
            float prob = dets[i].prob[class_id];
            if (prob > thresh_calc_avg_iou) {
                detections_count++;
                detections = (box_prob*)realloc(detections, detections_count * sizeof(box_prob));
                detections[detections_count - 1].b = dets[i].bbox;
                detections[detections_count - 1].p = prob;
                //detections[detections_count - 1].image_index = image_index;
                detections[detections_count - 1].class_id = class_id;
                detections[detections_count - 1].truth_flag = 0;
                detections[detections_count - 1].unique_truth_index = -1;

                int truth_index = -1;
                float max_iou = 0;
                for(j = 0; j < num_labels; j++){
                    box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                    float current_iou = box_iou(dets[i].bbox, t);
                    if (current_iou > iou_thresh && class_id == truth[j].id) {
                        if (current_iou > max_iou) {
                            max_iou = current_iou;
                            truth_index = j;
                        }
                    }
                }
                if(truth_index > -1){
                    detections[detections_count - 1].truth_flag = 1;
                    detections[detections_count - 1].unique_truth_index = truth_index;
                }
            }
        }
    }
    //printf("detections_count: %d\n", detections_count);
    qsort(detections, detections_count, sizeof(box_prob), detections_comparator);

    typedef struct {
        double precision;
        double recall;
        int tp, fp, fn;
    } pr_t;

    // for PR-curve
    pr_t** pr = (pr_t**)calloc(classes, sizeof(pr_t*));
    for (i = 0; i < classes; ++i) {
        pr[i] = (pr_t*)calloc(detections_count, sizeof(pr_t));
    }
    //printf("\n detections_count = %d, truth_count = %d  \n", detections_count, num_labels);

    int* detection_per_class_count = (int*)calloc(classes, sizeof(int));
    for (j = 0; j < detections_count; ++j) {
        detection_per_class_count[detections[j].class_id]++;
    }

    int* truth_flags = (int*)calloc(num_labels, sizeof(int));

    int rank;
    for (rank = 0; rank < detections_count; ++rank) {
        if (rank > 0) {
            int class_id; 
            for (class_id = 0; class_id < classes; ++class_id) {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }
        box_prob d = detections[rank];
        // if (detected && isn't detected before)
        if (d.truth_flag == 1) {
            if (truth_flags[d.unique_truth_index] == 0)
            {   
                truth_flags[d.unique_truth_index] = 1;
                pr[d.class_id][rank].tp++;    // true-positive
            } else
                pr[d.class_id][rank].fp++;
        }
        else {
            pr[d.class_id][rank].fp++;    // false-positive
        }

        for (i = 0; i < classes; ++i)
        {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_classes_count[i] - tp;    // false-negative = objects - true-positive
            pr[i][rank].fn = fn;

            if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
            else pr[i][rank].precision = 0;

            if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
            else pr[i][rank].recall = 0;

            if (rank == (detections_count - 1) && detection_per_class_count[i] != (tp + fp)) {    // check for last rank
                    printf(" class_id: %d - detections = %d, tp+fp = %d, tp = %d, fp = %d \n", i, detection_per_class_count[i], tp+fp, tp, fp);
            }
        }
    }
    if(0){
        for(i = 0; i < classes; i++){
            for(j = 0; j < detections_count; j++){
                printf("tp: %d, fp: %d, fn: %d, prec: %f, recall: %f\n", pr[i][j].tp, pr[i][j].fp, pr[i][j].fn, pr[i][j].precision, pr[i][j].recall);
            }
        }
    }
            
    double mean_average_precision = 0;
    for (i = 0; i < classes; ++i) {
        double avg_precision = 0;
        double last_recall = pr[i][detections_count - 1].recall;
        double last_precision = pr[i][detections_count - 1].precision;
        for (rank = detections_count - 2; rank >= 0; --rank)
        {
            double delta_recall = last_recall - pr[i][rank].recall;
            last_recall = pr[i][rank].recall;

            if (pr[i][rank].precision > last_precision) {
                last_precision = pr[i][rank].precision;
            }
            avg_precision += delta_recall * last_precision;
        }
        mean_average_precision += avg_precision;
    }
    mean_average_precision = mean_average_precision / classes;

    free(truth_flags);
    for (i = 0; i < classes; ++i) {
        free(pr[i]);
    }
    free(pr);
    free(detections);
    free(truth_classes_count);
    free(detection_per_class_count);

    return mean_average_precision;
}

void print_avf_log(attack_args *a, FILE *fp)
{
    int i;
    fprintf(fp, "avf: \n");
    for(i = 0; i < a->topk; i++)
        fprintf(fp, "%f ", a->avf[i]);
    fprintf(fp, "\n");
}
