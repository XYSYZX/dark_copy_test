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

    //weight
    topk = a->topk_weights;
    a->grads_loc_weights = make_2d_array_int(layer_num, topk * iter);
    a->mloss_loc_weights = make_2d_array_int(layer_num, topk);
    a->mloss_weights = make_2d_array_float(layer_num, topk * fb_len);
    a->macc_weights = make_2d_array_float(layer_num, topk * fb_len);
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
    a->mloss_biases = make_2d_array_float(layer_num, topk * fb_len);
    a->macc_biases = make_2d_array_float(layer_num, topk * fb_len);
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
}

void free_attack_args(attack_args a)
{
    int layer_num = a.layer_num;
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
    }
    {
        if(a.grads_loc_weights) free_2d_array_int(a.grads_loc_weights, layer_num);
        if(a.mloss_loc_weights) free_2d_array_int(a.mloss_loc_weights, layer_num);
        if(a.mloss_weights) free_2d_array_float(a.mloss_weights, layer_num);
        if(a.macc_weights) free_2d_array_float(a.macc_weights, layer_num);
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
        if(a.macc_biases) free_2d_array_float(a.macc_biases, layer_num);
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
        if(a.macc_outputs) free_2d_array_float(a.macc_outputs, layer_num);
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
    set_batch_network(net, args.n);

    data val, buffer;
    args.d = &buffer;

    pthread_t load_thread = load_data(args);
    attack->seen_img += args.n;

    for(i = 0; i < iter; i++){
        attack->iter_idx = i;
        pthread_join(load_thread, 0);
        val = buffer;
        fprintf(stderr, "images: %d\n", attack->seen_img);
        loss += network_predict_search(net, val) / iter;
        for(j = 0; j < attack->layer_num; j++){
            //printf("grad: %x, gpu: %x\n", attack->grads[j], attack->grads_gpu[j]);
            attack->layer_idx = j;
            get_topk_grad(attack);
        }
        load_thread = load_data(args);
        attack->seen_img += args.n;
    }
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
        #ifdef GPU
            cuda_pull_array(a.grads_gpu[a.layer_idx], a.grads[a.layer_idx], a.len[a.layer_idx]);
        #endif
            //bit_flip_attacker(a, a.x[a.layer_idx], a.grads[a.layer_idx], a.mloss_loc[a.layer_idx][offset]);
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
    int idx = a.mloss_loc[a.layer_idx][a.k_idx];
    float *x = a.x[a.layer_idx];
    int bit_idx = a.bit_idx;
    inject_noise_float_onebit(x, idx, bit_idx);
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
    /*
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
    */
    free_2d_array_int(mloss_freq_input, 1);
    free_2d_array_int(mloss_freq_weight, layer_num);
    free_2d_array_int(mloss_freq_bias, layer_num);
    free_2d_array_int(mloss_freq_output, layer_num);
}

void get_avf(network *net, load_args args, int type)
{
    attack_args *a = net->attack;
    float avg_loss = 0;
    float avg_acc = 0;
    int layer_num = net->n;

    int i, j, k, m;

    //if(a->sign_attack) args.n = net->batch;
    //int iter = args.m / args.n;

    data val, buffer;
    args.d = &buffer;

    set_batch_network(net, args.n);

    int iter = args.m / args.n;
    a->iter = iter;

    const float nms = 0.45;
    const float thresh = 0.5;
    const float iou_thresh = 0.5;
    const float thresh_calc_avg_iou = 0.1;
    int classes = net->layers[net->n-1].classes;

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
            a->macc = a->macc_inputs;
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
            a->macc = a->macc_weights;
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
            a->macc = a->macc_biases;
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
            a->macc = a->macc_outputs;
            a->mloss_loc = a->mloss_loc_outputs;
            a->avf = a->avf_outputs;
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
    if(count_boxes <= args.num_boxes) args.num_boxes = count_boxes;

    pthread_t load_thread = load_data_in_thread(args);
    a->seen_img += args.n;

    for(i = 0; i < iter; i++){
        pthread_join(load_thread, 0);
        val = buffer;

        a->iter_idx = i;
        fprintf(stderr, "images: %d\n", a->seen_img);
        if(0){
            printf("box_label id: \n");
            for(int x = 0; x < args.num_boxes; x++){
                printf("%d\t", args.boxes[x].id);
            }
            printf("\n");
        }
        avg_loss += network_predict_search(net, val) / iter;
        int nboxes = 0;
        detection *dets = get_network_boxes(net, net->w, net->h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, classes, nms);

        avg_acc += cal_map(net, dets, args.boxes, nboxes, args.num_boxes, iou_thresh, thresh_calc_avg_iou) / iter;
        free_detections(dets, nboxes);
        //printf("avg_loss: %f, avg_acc: %f\n", avg_loss, avg_acc);

        for(j = 0; j < a->layer_num; j++){
            a->layer_idx = j;
            if(a->len[j] == 0) continue;
            for(k = 0; k < a->topk; k++){
                a->k_idx = k;
                for(m = 0; m < a->fb_len; m++){
                    a->bit_idx = a->flipped_bit[m];
                    if(a->a_weight || a->a_bias){
                        single_attack(net);
                    }

                    a->mloss[j][a->fb_len*k+m] += network_predict_attack(net, val) / iter;
                    int nboxes = 0;
                    detection *dets = get_network_boxes(net, net->w, net->h, thresh, .5, 0, 1, &nboxes); 
                    if (nms) do_nms_sort(dets, nboxes, classes, nms);
                    if(0){
                        printf("detection objectness: \n");
                        for(int x = 0; x < nboxes; x++){
                            printf("%f ", dets[x].objectness);
                        }
                        printf("\n");
                    }

                    a->macc[j][a->fb_len*k+m] += cal_map(net, dets, args.boxes, nboxes, args.num_boxes, iou_thresh, thresh_calc_avg_iou) / iter;
                    free_detections(dets, nboxes);
                    //printf("layer: %d, k: %d, bit: %d, loss: %f, acc: %f\n", j, k, a->bit_idx, a->mloss[j][a->fb_len*k+m], a->macc[j][a->fb_len*k+m]);

                    if(a->a_weight || a->a_bias){
                        single_attack(net);
                    }
                    //printf("single: %f\n", attack->mloss[j][i*attack->topk + k]);
                }
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
        if(count_boxes <= args.num_boxes) args.num_boxes = count_boxes;

        load_thread = load_data_in_thread(args);
        a->seen_img += args.n;
    }

    a->loss_thresh = avg_loss;
    a->acc_thresh = avg_acc;
    cal_avf(a);
    printf("type: %d, avf: \n", type);
    print_2d_array_float(a->avf, a->layer_num, a->topk);
    if(type == 0) a->a_input = 0;
    if(type == 1) a->a_weight = 0;
    if(type == 2) a->a_bias = 0;
    if(type == 3) a->a_output = 0;
    a->seen_img = 0;
}

void cal_avf(attack_args *a)
{
    int i, j, m;
    float max_loss = 0;
    float loss = 0;
    float **losses = make_2d_array_float(a->layer_num, a->topk);
    float max_acc = 0;
    float acc = 0;
    float **accs = make_2d_array_float(a->layer_num, a->topk);

    float **avf = a->avf;
    float **mloss = a->mloss;
    float **macc = a->macc;

    for(i = 0; i < a->layer_num; i++){
        if(a->len[i] == 0) continue;
        for(j = 0; j < a->topk; j++){
            for(m = 0; m < a->fb_len; m++){
                if(mloss[i][j*a->fb_len+m] > a->loss_thresh) loss++;
                if(macc[i][j*a->fb_len+m] > a->acc_thresh) acc++;
            ;}
            losses[i][j] = loss;
            max_loss = loss > max_loss ? loss: max_loss;
            loss = 0;
            accs[i][j] = acc;
            max_acc = acc > max_acc? acc: max_acc;
            acc = 0;
        }
    }
    printf("max loss: %f, max acc: %f\n", max_loss, max_acc);
    for(i = 0; i < a->layer_num; i++){
        if(a->len[i] == 0) continue;
        for(j = 0; j < a->topk; j++){
            avf[i][j] = (losses[i][j] / max_loss) * a->alpha + (max_acc - accs[i][j])/max_acc * (1 - a->alpha) ;
        }
    }
    print_2d_array_float(mloss, a->layer_num, a->topk*a->fb_len);
    print_2d_array_float(macc, a->layer_num, a->topk*a->fb_len);
    //print_2d_array_float(losses, a->layer_num, a->topk);
    //print_2d_array_float(accs, a->layer_num, a->topk);
    free_2d_array_float(losses, a->layer_num);
    free_2d_array_float(accs, a->layer_num);
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

