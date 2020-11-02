#include <float.h>
#include "bit_attack.h"
#include "sort.h"

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

void set_2d_array_float(float **x, int n, int m, float val)
{
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            x[i][j] = val;
}

void set_2d_array_int(int **x, int n, int m, int val)
{
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            x[i][j] = val;
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

    //filter
    if(a->a_filter){
        topk = a->topk_flt;
        a->grads_loc_flt = make_2d_array_int(layer_num, topk * iter);
        a->grads_val_flt = make_2d_array_float(layer_num, topk * iter);
        if(a->worst){
            a->grads_loc_flt_min = make_2d_array_int(layer_num, topk * iter);
            a->grads_val_flt_min = make_2d_array_float(layer_num, topk * iter);
        }

        a->mloss_loc_flt = a->worst? make_2d_array_int(2, topk*2): make_2d_array_int(2, topk);
        a->mloss_flt = a->worst? (float *)calloc(topk*2 * fb_len, sizeof(float)): (float *)calloc(topk * fb_len, sizeof(float));
        a->macc_flt = a->worst? (float *)calloc(topk*2 * fb_len, sizeof(float)): (float *)calloc(topk * fb_len, sizeof(float));
        a->avf_flt = a->worst? (float *)calloc(topk*2, sizeof(float)): (float *)calloc(topk, sizeof(float));

        set_2d_array_float(a->grads_val_flt, layer_num, topk*iter, -1);
        set_2d_array_int(a->grads_loc_flt, layer_num, topk*iter, -1);
        if(a->worst){
            set_2d_array_float(a->grads_val_flt_min, layer_num, topk*iter, FLT_MAX);
            set_2d_array_int(a->grads_loc_flt_min, layer_num, topk*iter, -1);
        }

        a->flt_len = (int *)calloc(net->n, sizeof(int));
        a->flt_size = (int *)calloc(net->n, sizeof(int));
        //int count = 0;
        for(i = 0; i < net->n; i++){
            if(net->layers[i].type == CONVOLUTIONAL){
                layer l = net->layers[i];
                a->flt_len[i] = l.n;
                a->flt_size[i] = l.c*l.size*l.size; //c是input的通道数
                //count += l.n;
            }
        }
        
        a->topks_flt = (int *)calloc(net->n, sizeof(int));
        for(i = 0; i < net->n; i++){
            if(a->flt_len[i] > a->topk_flt) a->topks_flt[i] = a->topk_flt;
            else a->topks_flt[i] = a->flt_len[i];
        }

        a->get_topk_grad = get_topk_grad_flt;
        a->get_max_loss = get_max_loss_flt;
        //a->get_avf = get_avf_wb;
        a->single_attack = single_attack_flt;
    }
   

    //layer
    else if(a->a_layer){
        topk = a->topk_l;
        for(i = 0; i < net->n; i++) a->all_weight += a->weights_len[i];
        for(i = 0; i < net->n; i++) a->all_bias += a->biases_len[i];

        int allk = a->all_weight / a->dist_fac;

        a->grads_val_l = (float *)calloc(allk, sizeof(float));
        a->grads_loc_l = (int *)calloc(allk, sizeof(int));
        a->grads_loc_layer_l = (int *)calloc(allk, sizeof(int));

        a->mloss_loc_l = make_2d_array_int(2, topk);
        a->mloss_l = (float *)calloc(topk*fb_len, sizeof(float));
        a->macc_l = (float *)calloc(topk*fb_len, sizeof(float));
        a->avf_l = (float *)calloc(topk, sizeof(float));

        a->get_topk_grad = get_topk_grad_l;
        a->get_max_loss = get_max_loss_l;
        //a->get_avf = get_avf_wb;
        a->single_attack = single_attack_l;
    }
        
    else{
        //weight
        topk = a->topk_weight;
        a->grads_loc_weights = make_2d_array_int(layer_num, topk * iter);
        a->grads_val_weights = make_2d_array_float(layer_num, topk * iter);
        if(a->worst){
            a->grads_loc_weights_min = make_2d_array_int(layer_num, topk * iter);
            a->grads_val_weights_min = make_2d_array_float(layer_num, topk * iter);
        }

        a->mloss_loc_weights = a->worst? make_2d_array_int(2, topk*2): make_2d_array_int(2, topk);
        a->mloss_weights = a->worst? (float *)calloc(topk*2 * fb_len, sizeof(float)): (float *)calloc(topk * fb_len, sizeof(float));
        a->macc_weights = a->worst? (float *)calloc(topk*2 * fb_len, sizeof(float)): (float *)calloc(topk * fb_len, sizeof(float));
        a->avf_weights = a->worst? (float *)calloc(topk*2, sizeof(float)): (float *)calloc(topk*2, sizeof(float));

        set_2d_array_float(a->grads_val_weights, layer_num, topk*iter, -1);
        set_2d_array_int(a->grads_loc_weights, layer_num, topk*iter, -1);
        if(a->worst){
            set_2d_array_float(a->grads_val_weights_min, layer_num, topk*iter, FLT_MAX);
            set_2d_array_int(a->grads_loc_weights_min, layer_num, topk*iter, -1);
        }

        a->topk_weights = (int *)calloc(net->n, sizeof(int));
        for(i = 0; i < net->n; i++){
            if(a->weights_len[i] > a->topk_weight) a->topk_weights[i] = a->topk_weight;
            else a->topk_weights[i] = a->weights_len[i];
        }

        //bias
        topk = a->topk_bias;
        a->grads_loc_biases = make_2d_array_int(layer_num, topk * iter);
        a->grads_val_biases = make_2d_array_float(layer_num, topk * iter);
        if(a->worst){
            a->grads_loc_biases_min = make_2d_array_int(layer_num, topk * iter);
            a->grads_val_biases_min = make_2d_array_float(layer_num, topk * iter);
        }

        a->mloss_loc_biases = a->worst? make_2d_array_int(2, topk*2): make_2d_array_int(2, topk);
        a->mloss_biases = a->worst? (float *)calloc(topk*2 * fb_len, sizeof(float)): (float *)calloc(topk * fb_len, sizeof(float));
        a->macc_biases = a->worst? (float *)calloc(topk*2 * fb_len, sizeof(float)): (float *)calloc(topk * fb_len, sizeof(float));
        a->avf_biases = a->worst? (float *)calloc(topk*2 * fb_len, sizeof(float)): (float *)calloc(topk, sizeof(float));

        set_2d_array_float(a->grads_val_biases, layer_num, topk*iter, -1);
        set_2d_array_int(a->grads_loc_biases, layer_num, topk*iter, -1);
        if(a->worst){
            set_2d_array_float(a->grads_val_biases_min, layer_num, topk*iter, FLT_MAX);
            set_2d_array_int(a->grads_loc_biases_min, layer_num, topk*iter, -1);
        }

        a->topk_biases = (int *)calloc(net->n, sizeof(int));
        for(i = 0; i < net->n; i++){
            if(a->biases_len[i] > a->topk_bias) a->topk_biases[i] = a->topk_bias;
            else a->topk_biases[i] = a->biases_len[i];
        }
        
        a->get_topk_grad = get_topk_grad_wb;
        a->get_max_loss = get_max_loss_wb;
        //a->get_avf = get_avf_wb;
        a->single_attack = single_attack_wb;
    }
}

void free_attack_args(attack_args a)
{
    int layer_num = a.layer_num;
    {
        if(a.grads_loc_flt) free_2d_array_int(a.grads_loc_flt, layer_num);
        if(a.grads_val_flt) free_2d_array_float(a.grads_val_flt, layer_num);
        if(a.grads_loc_flt_min) free_2d_array_int(a.grads_loc_flt_min, layer_num);
        if(a.grads_val_flt_min) free_2d_array_float(a.grads_val_flt_min, layer_num);
        if(a.mloss_loc_flt) free_2d_array_int(a.mloss_loc_flt, 2);
        if(a.mloss_flt) free(a.mloss_flt);
        if(a.macc_flt) free(a.macc_flt);
        if(a.avf_flt) free(a.avf_flt);
        if(a.flt_len) free(a.flt_len);
        if(a.flt_size) free(a.flt_len);
        //if(a.flt_layer_idx) free(a.flt_len);
    }
    {
        if(a.grads_val_l) free(a.grads_val_l);
        if(a.grads_loc_l) free(a.grads_loc_l);
        if(a.grads_loc_layer_l) free(a.grads_loc_layer_l);
        if(a.mloss_loc_l) free_2d_array_int(a.mloss_loc_l, 2);
        if(a.mloss_l) free(a.mloss_l);
        if(a.macc_l) free(a.macc_l);
        if(a.avf_l) free(a.avf_l);
    }
    {
        if(a.grads_loc_weights) free_2d_array_int(a.grads_loc_weights, layer_num);
        if(a.grads_val_weights) free_2d_array_float(a.grads_val_weights, layer_num);
        if(a.grads_loc_weights_min) free_2d_array_int(a.grads_loc_weights_min, layer_num);
        if(a.grads_val_weights_min) free_2d_array_float(a.grads_val_weights_min, layer_num);
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
        if(a.grads_loc_biases_min) free_2d_array_int(a.grads_loc_biases_min, layer_num);
        if(a.grads_val_biases_min) free_2d_array_float(a.grads_val_biases_min, layer_num);
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
}

void attack_data(network *net, load_args args, load_args val_args)
{
    double loss = 0;
    int i, j;

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
            attack->get_topk_grad(attack);
        }
        load_thread = load_data_in_thread(args);
        free_data(val);
        attack->seen_img += args.n;
        args.paths += args.n;
        fprintf(stderr, "get topk time: %f\n", what_time_is_it_now() - time);
    }
    attack->get_max_loss(attack);
    //exit(0);
    attack->seen_img = 0;
    FILE *avf_fp = fopen(attack->avf_log, "w+");
    if(!avf_fp){
        printf("no avg file!");
        return;
    }
    if(!attack->a_filter && !attack->a_layer) for(i = 0; i < 2; get_avf(net, val_args, i++, avf_fp));
    else if(attack->a_filter) get_avf(net, val_args, 2, avf_fp);
    else get_avf(net, val_args, 3, avf_fp);

    fclose(avf_fp);
    free_attack_args(*attack);
}

void get_topk_grad_wb(attack_args *a)
{
    int j = a->layer_idx;
    int i = a->iter_idx;
    
    if(a->weight_grads[j] && a->weight_grads_gpu[j]){
        get_topk(a->weight_grads[j], a->weight_grads_gpu[j], a->weights_len[j], a->grads_loc_weights[j]+i*a->topk_weight, a->grads_val_weights[j]+i*a->topk_weight, a->topk_weights[j], 0);
    }
    if(a->bias_grads[j] && a->bias_grads_gpu[j]){
        get_topk(a->bias_grads[j], a->bias_grads_gpu[j], a->biases_len[j], a->grads_loc_biases[j]+i*a->topk_bias, a->grads_val_biases[j]+i*a->topk_bias, a->topk_biases[j], 0);
    }
    if(a->worst){
        if(a->weight_grads[j] && a->weight_grads_gpu[j]){
            get_topk(a->weight_grads[j], a->weight_grads_gpu[j], a->weights_len[j], a->grads_loc_weights_min[j]+i*a->topk_weight, a->grads_val_weights_min[j]+i*a->topk_weight, a->topk_weights[j], 1);
        }
        if(a->bias_grads[j] && a->bias_grads_gpu[j]){
            get_topk(a->bias_grads[j], a->bias_grads_gpu[j], a->biases_len[j], a->grads_loc_biases_min[j]+i*a->topk_bias, a->grads_val_biases_min[j]+i*a->topk_bias, a->topk_biases[j], 1);
        }
    }
}

void get_topk_grad_flt(attack_args *a)
{
    int j = a->layer_idx;
    int i = a->iter_idx;
    int flt_len = a->flt_len[j];
    int *grads_loc, *grads_loc_min;
    float *grads_val, *grads_val_min; 
    if(flt_len){
        //int flt_size = a->flt_size[j];
        grads_loc = a->grads_loc_flt[j];
        grads_val = a->grads_val_flt[j];
        if(a->worst){
            grads_loc_min = a->grads_loc_flt_min[j];
            grads_val_min = a->grads_val_flt_min[j];
        }

        float *flt_grad = (float *)calloc(flt_len, sizeof(float));
        cal_grad_flt(a, flt_grad);
        get_topk(flt_grad, 0, flt_len, grads_loc+i*a->topk_flt, grads_val+i*a->topk_flt, a->topks_flt[j], 0);
        if(a->worst){
            get_topk(flt_grad, 0, flt_len, grads_loc_min+i*a->topk_flt, grads_val_min+i*a->topk_flt, a->topks_flt[j], 1);
        }
        free(flt_grad);
    }
}

void get_topk_grad_l(attack_args *a)
{
    int layer_idx = a->layer_idx;
    double time;
    if(layer_idx == 0){
        int layer_num = a->layer_num;
        float *all_weight_grad = (float *)calloc(a->all_weight, sizeof(float));
        int *all_weight_loc = (int *)calloc(a->all_weight, sizeof(int));
        int *all_weight_layer_loc = (int *)calloc(a->all_weight, sizeof(int));
        int offset = 0;
        for(int i = 0; i < layer_num; i++){
            if(a->weight_grads_gpu[i]){
                abs_gpu(a->weight_grads_gpu[i], a->weight_grads_gpu[i], a->weights_len[i]);
                cuda_pull_array(a->weight_grads_gpu[i], a->weight_grads[i], a->weights_len[i]);
            }
            else abs_cpu(a->weight_grads[i], a->weight_grads[i], a->weights_len[i]);
            for(int j = 0; j < a->weights_len[i]; j++){
                all_weight_layer_loc[offset+j] = i;
                all_weight_loc[offset+j] = j;
                all_weight_grad[offset+j] = a->weight_grads[i][j];
            }
            offset += a->weights_len[i];
        }
        time = what_time_is_it_now();
        heapsort_with_layer(all_weight_grad, all_weight_loc, all_weight_layer_loc, a->all_weight, 0);
        //qsort_with_layer(all_weight_grad, all_weight_loc, all_weight_layer_loc, 0, a->all_weight-1, 0);
        printf("finish sort, time use: %f\n", what_time_is_it_now() - time);
        int allk = a->all_weight/a->dist_fac;
        get_same_dist(all_weight_grad, all_weight_loc, all_weight_layer_loc, a->all_weight, a->grads_val_l, a->grads_loc_l, a->grads_loc_layer_l, allk);
        printf("big layer value and loc\n");
        for(int i = 0; i < allk; i++) printf("%f ", a->grads_val_l[i]);
        printf("\n");
        for(int i = 0; i < allk; i++) printf("%d ", a->grads_loc_l[i]);
        printf("\n");
        for(int i = 0; i < allk; i++) printf("%d ", a->grads_loc_layer_l[i]);
        printf("\n");
        free(all_weight_loc);
        free(all_weight_grad);
        free(all_weight_layer_loc);
    }
}

void cal_grad_flt(attack_args *a, float *flt_grad)
{
    int i = a->layer_idx;
    float *weight_grad = a->weight_grads[i];
    float *bias_grad = a->bias_grads[i];
    float *weight_grad_gpu = a->weight_grads_gpu[i];
    float *bias_grad_gpu = a->bias_grads_gpu[i];
    int w_len = a->weights_len[i];
    int b_len = a->biases_len[i];
    int flt_len = a->flt_len[i];
    int flt_size = a->flt_size[i];

    if(weight_grad_gpu){
        abs_gpu(weight_grad_gpu, weight_grad_gpu, w_len);
        cuda_pull_array(weight_grad_gpu, weight_grad, w_len);
    }
    else abs_cpu(weight_grad, weight_grad, w_len);
    if(bias_grad_gpu){
        abs_gpu(bias_grad_gpu, bias_grad_gpu, b_len);
        cuda_pull_array(bias_grad_gpu, bias_grad, b_len);
    }
    else abs_cpu(bias_grad, bias_grad, b_len);

    for(int i = 0; i < flt_len; i++){
        float tmp = 0;
        for(int j = 0; j < flt_size; j++){
            tmp += weight_grad[flt_size*i + j];
        }
        tmp += bias_grad[i];
        flt_grad[i] = tmp / (flt_size+1);
    }
}

void single_attack_l(network *net)
{
    attack_args a = *(net->attack);
    int allk = a.all_weight / a.dist_fac;
    int step = allk / a.topk_l;
    int start = step * a.k_idx;
    if ((start+step) >=a.all_weight) return;
    for(int i = 0; i < step; i++){
        int layer_idx = a.grads_loc_layer_l[start+i];
        int idx = a.grads_loc_l[start+i];
        int bit_idx = a.bit_idx;
    #ifdef GPU
        bit_flip_attacker_gpu(a.x_gpu[layer_idx], idx, bit_idx);
        cudaDeviceSynchronize();
    #else
        bit_flip_attacker(a.x[layer_idx], idx, bit_idx);
    #endif
    }
}

void single_attack_wb(network *net)
{
    attack_args a = *(net->attack);
    int layer_idx = a.mloss_loc[0][a.k_idx];
    int idx = a.mloss_loc[1][a.k_idx];
    int bit_idx = a.bit_idx;

    if(a.sign_attack){
        if(a.reverse == 1){
        #ifdef GPU
            sign_attacker_gpu(a.x_gpu[layer_idx], a.grads_gpu[layer_idx], idx, a.epsilon);
        #else
            sign_attacker(a.x[layer_idx], a.grads[layer_idx], idx, a.epsilon);
        #endif
        }
        else{
        #ifdef GPU
            sign_delete_gpu(a.x_gpu[layer_idx], a.grads_gpu[layer_idx], idx, a.epsilon);
        #else
            sign_delete(a.x[layer_idx], a.grads[layer_idx], idx, a.epsilon);
        #endif
        }
    }
    else{
    #ifdef GPU
        bit_flip_attacker_gpu(a.x_gpu[layer_idx], idx, bit_idx);
    #else
        bit_flip_attacker(a.x[layer_idx], idx, bit_idx);
    #endif
    }
    //printf("finish bit flip\n");
    if(a.a_weight || a.a_bias) {
        net->attack->reverse *= -1;
    }
}

void single_attack_flt(network *net)
{
    attack_args a = *(net->attack);
    int idx;
    int layer_idx = a.mloss_loc[0][a.k_idx];
    int flt_idx = a.mloss_loc[1][a.k_idx];
    int bit_idx = a.bit_idx;

    if(a.sign_attack){
        if(a.reverse == 1){
            for(int i = 0; i < a.flt_size[layer_idx]; i++){ 
                idx = flt_idx * a.flt_size[layer_idx] + i;
            #ifdef GPU
                sign_attacker_gpu(a.weights_gpu[layer_idx], a.weight_grads_gpu[layer_idx], idx, a.epsilon);
            }
            sign_attacker_gpu(a.biases_gpu[layer_idx], a.bias_grads_gpu[layer_idx], flt_idx, a.epsilon);
            #else
                sign_attacker(a.weights[layer_idx], a.weight_grads[layer_idx], idx, a.epsilon);
            }
            sign_attacker(a.biases[layer_idx], a.bias_grads[layer_idx], flt_idx, a.epsilon);
            #endif
        }
        else{
            for(int i = 0; i < a.flt_size[layer_idx]; i++){ 
                idx = flt_idx * a.flt_size[layer_idx] + i;
            #ifdef GPU
                sign_delete_gpu(a.weights_gpu[layer_idx], a.weight_grads_gpu[layer_idx], idx, a.epsilon);
            }
            sign_delete_gpu(a.biases_gpu[layer_idx], a.bias_grads_gpu[layer_idx], flt_idx, a.epsilon);
            #else
                sign_delete(a.weights[layer_idx], a.weight_grads[layer_idx], idx, a.epsilon);
            }
            sign_delete(a.biases[layer_idx], a.bias_grads[layer_idx], flt_idx, a.epsilon);
            #endif
        }
    }

    else{
        for(int i = 0; i < a.flt_size[layer_idx]; i++){
            idx = flt_idx * a.flt_size[layer_idx] + i;
        #ifdef GPU
            bit_flip_attacker_gpu(a.weights_gpu[layer_idx], idx, bit_idx);
            //cudaDeviceSynchronize();

        }
        bit_flip_attacker_gpu(a.biases_gpu[layer_idx], flt_idx, bit_idx);
        //cudaDeviceSynchronize();
        #else
            bit_flip_attacker(a.weights[layer_idx], idx, bit_idx);
        }
        bit_flip_attacker(a.biases[layer_idx], flt_idx, bit_idx);
        #endif
    }
    //printf("\nfinish a single attack\n");
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

void sign_attacker(float *x, float *grad, int idx, float epsilon)
{
    float x_sign = sign(grad[idx]);
    x[idx] -= epsilon * x_sign;
}

void sign_delete(float *x, float *grad, int idx, float epsilon)
{
    float x_sign = sign(grad[idx]);
    x[idx] += epsilon * x_sign;
}

void bit_flip_attacker(float *x, int idx, int bit_idx)
{
    inject_noise_float_onebit(x, idx, bit_idx);
}

void get_max_loss_l(attack_args *a)
{
    printf("do nothing temporary!\n");
}

void get_max_loss_flt(attack_args *a)
{
    int i, j;
    int layer_num = a->layer_num;
    int iter = a->iter;
    int topk = a->topk_flt;
    int *topks = a->topks_flt;
    int *flt_len = a->flt_len;

    float *maxloss_val_flt = (float *)calloc(layer_num * topk, sizeof(float));
    int *maxloss_loc_flt = (int *)calloc(layer_num * topk, sizeof(int));
    float *minloss_val_flt = (float *)calloc(layer_num * topk, sizeof(float));
    int *minloss_loc_flt = (int *)calloc(layer_num * topk, sizeof(int));
    for(i = 0; i < layer_num*topk; i++) minloss_val_flt[i] = FLT_MAX;
    for(i = 0; i < layer_num*topk; i++) maxloss_val_flt[i] = -1;

    int count = 0;
    for(i = 0; i < layer_num; i++){
        if(flt_len[i]){
            count = i * topk;
            if(0){
                int *count_flt = (int *)calloc(flt_len[i], sizeof(int));
                float *val_flt = (float *)calloc(flt_len[i], sizeof(float));
                int len = iter * topk;
                for(j = 0; j < len ; j++){
                    if(a->grads_loc_flt[i][j] >= 0){
                        count_flt[a->grads_loc_flt[i][j]] += 1;
                        val_flt[a->grads_loc_flt[i][j]] += a->grads_val_flt[i][j];
                    }
                }
                for(j = 0; j < len; j++){
                    if(a->grads_loc_flt[i][j] >= 0){
                        val_flt[a->grads_loc_flt[i][j]] /= count_flt[a->grads_loc_flt[i][j]];
                    }
                }
                get_topk_int(count_flt, val_flt, flt_len[i], topks[i], maxloss_loc_flt+count, maxloss_val_flt+count, 0);

                if(a->worst){
                    for(j = 0; j < flt_len[i]; j++){
                        count_flt[j] = 0;
                        val_flt[j] = 0;
                    }
                    for(j = 0; j < len ; j++){
                        if(a->grads_loc_flt_min[i][j] >= 0){
                            count_flt[a->grads_loc_flt_min[i][j]] += 1;
                            val_flt[a->grads_loc_flt_min[i][j]] += a->grads_val_flt_min[i][j];
                        }
                    }
                    for(j = 0; j < len; j++){
                        if(a->grads_loc_flt_min[i][j] >= 0){
                            val_flt[a->grads_loc_flt_min[i][j]] /= count_flt[a->grads_loc_flt_min[i][j]];
                        }
                    }
                    get_topk_int(count_flt, val_flt, flt_len[i], topks[i], minloss_loc_flt+count, minloss_val_flt+count, 1);
                }
                free(count_flt);
                free(val_flt);
            }
            
            if(1){
                get_topk_float(a->grads_val_flt[i], a->grads_loc_flt[i], iter*topk, topks[i], maxloss_val_flt + count, maxloss_loc_flt + count, 0);
                if(a->worst) get_topk_float(a->grads_val_flt_min[i], a->grads_loc_flt_min[i], iter*topk, topks[i], minloss_val_flt + count, minloss_loc_flt + count, 1);
            }
        }
    }
    int offset = topk;
    get_topk_with_layer(maxloss_val_flt, maxloss_loc_flt, topk*layer_num, topk, a->mloss_loc_flt[0], a->mloss_loc_flt[1], 0);
    if(a->worst){
        get_topk_with_layer(minloss_val_flt, minloss_loc_flt, topk*layer_num, topk, a->mloss_loc_flt[0]+offset, a->mloss_loc_flt[1]+offset, 1);
    }

    printf("max/min layer loc of filter\n");
    for(i = 0; i < a->topk_flt; i++) printf("%d ", a->mloss_loc_flt[0][i]);
    printf("\n");
    for(i = 0; i < a->topk_flt; i++) printf("%d ", a->mloss_loc_flt[1][i]);
    printf("\n");

    if(a->worst){
        for(i = 0; i < a->topk_flt; i++) printf("%d ", a->mloss_loc_flt[0][i+offset]);
        printf("\n");
        for(i = 0; i < a->topk_flt; i++) printf("%d ", a->mloss_loc_flt[1][i+offset]);
        printf("\n");
    }

    free(maxloss_val_flt);
    free(maxloss_loc_flt);
    free(minloss_val_flt);
    free(minloss_loc_flt);
}

void get_max_loss_wb(attack_args *a)
{
    int i, j;
    int layer_num = a->layer_num;
    int iter = a->iter;
    //int *mloss_freq_weights = (int *)calloc(layer_num, a->topk_weights, sizeof(int));
    //int *mloss_freq_biases = (int *)calloc(layer_num, a->topk_biases, sizeof(int));
    float *maxloss_val_weights = (float *)calloc(layer_num * a->topk_weight, sizeof(float));
    float *maxloss_val_biases = (float *)calloc(layer_num * a->topk_bias, sizeof(float));
    int *maxloss_loc_weights = (int *)calloc(layer_num * a->topk_weight, sizeof(int));
    int *maxloss_loc_biases = (int *)calloc(layer_num * a->topk_bias, sizeof(int));
    float *minloss_val_weights = (float *)calloc(layer_num * a->topk_weight, sizeof(float));
    float *minloss_val_biases = (float *)calloc(layer_num * a->topk_bias, sizeof(float));
    int *minloss_loc_weights = (int *)calloc(layer_num * a->topk_weight, sizeof(int));
    int *minloss_loc_biases = (int *)calloc(layer_num * a->topk_bias, sizeof(int));
    
    for(i = 0; i < layer_num*a->topk_weight; i++) minloss_val_weights[i] = FLT_MAX;
    for(i = 0; i < layer_num*a->topk_bias; i++) minloss_val_biases[i] = FLT_MAX;
    for(i = 0; i < layer_num*a->topk_weight; i++) maxloss_val_weights[i] = -1;
    for(i = 0; i < layer_num*a->topk_bias; i++) maxloss_val_biases[i] = -1;

    int count = 0;

    for(i = 0; i < layer_num; i++){
        if(a->weights_len[i]){
            count = i * a->topk_weight;
            if(0){
                int *count_weight = (int *)calloc(a->weights_len[i], sizeof(int));
                float *val_weight = (float *)calloc(a->weights_len[i], sizeof(float));
                int len = iter * a->topk_weight;
                for(j = 0; j < len ; j++){
                    if(a->grads_loc_weights[i][j] >= 0){
                        count_weight[a->grads_loc_weights[i][j]] += 1;
                        val_weight[a->grads_loc_weights[i][j]] += a->grads_val_weights[i][j];
                    }
                }
                for(j = 0; j < len; j++){
                    if(a->grads_loc_weights[i][j] >= 0){
                        val_weight[a->grads_loc_weights[i][j]] /= count_weight[a->grads_loc_weights[i][j]];
                    }
                }
                get_topk_int(count_weight, val_weight, a->weights_len[i], a->topk_weights[i], maxloss_loc_weights+count, maxloss_val_weights+count, 0);

                if(a->worst){
                    for(int k = 0; k < a->weights_len[i]; k++){
                        count_weight[i] = 0;
                        val_weight[i] = 0;
                    }
                    for(j = 0; j < len ; j++){
                        if(a->grads_loc_weights_min[i][j] >= 0){
                            count_weight[a->grads_loc_weights_min[i][j]] += 1;
                            val_weight[a->grads_loc_weights_min[i][j]] += a->grads_val_weights_min[i][j];
                        }
                    }
                    for(j = 0; j < len; j++){
                        if(a->grads_loc_weights_min[i][j] >= 0){
                            val_weight[a->grads_loc_weights_min[i][j]] /= count_weight[a->grads_loc_weights_min[i][j]];
                        }
                    }
                    get_topk_int(count_weight, val_weight, a->weights_len[i], a->topk_weights[i], minloss_loc_weights+count, minloss_val_weights+count, 1);
                }
                free(count_weight);
                free(val_weight);
            }
            if(1){
                get_topk_float(a->grads_val_weights[i], a->grads_loc_weights[i], iter*a->topk_weight, a->topk_weights[i], maxloss_val_weights + count, maxloss_loc_weights + count, 0);
                if(a->worst) get_topk_float(a->grads_val_weights_min[i], a->grads_loc_weights_min[i], iter*a->topk_weight, a->topk_weights[i], minloss_val_weights + count, minloss_loc_weights + count, 1);
            }
        }

        if(a->biases_len[i]){
            count = i * a->topk_bias;
            if(0){
                int *count_bias = (int *)calloc(a->biases_len[i], sizeof(int));
                float *val_bias = (float *)calloc(a->biases_len[i], sizeof(float));
                int len = iter * a->topk_bias;
                for(j = 0; j < len; j++){
                    if(a->grads_loc_biases[i][j] >= 0){
                        count_bias[a->grads_loc_biases[i][j]] += 1;
                        val_bias[a->grads_loc_biases[i][j]] += a->grads_val_biases[i][j];
                    }
                }
                for(j = 0; j < len; j++){
                    if(a->grads_loc_biases[i][j] >= 0){
                        val_bias[a->grads_loc_biases[i][j]] /= count_bias[a->grads_loc_biases[i][j]];
                    }
                }
                get_topk_int(count_bias, val_bias, a->biases_len[i], a->topk_biases[i], maxloss_loc_biases+count, maxloss_val_biases+count, 0);

                if(a->worst){
                    for(int k = 0; k < a->biases_len[i]; k++){
                        count_bias[i] = 0;
                        val_bias[i] = 0;
                    }
                    for(j = 0; j < len; j++){
                        if(a->grads_loc_biases_min[i][j] >= 0){
                            count_bias[a->grads_loc_biases_min[i][j]] += 1;
                            val_bias[a->grads_loc_biases_min[i][j]] += a->grads_val_biases_min[i][j];
                        }
                    }
                    for(j = 0; j < len; j++){
                        if(a->grads_loc_biases_min[i][j] >= 0){
                            val_bias[a->grads_loc_biases_min[i][j]] /= count_bias[a->grads_loc_biases_min[i][j]];
                        }
                    }
                    get_topk_int(count_bias, val_bias, a->biases_len[i], a->topk_biases[i], minloss_loc_biases+count, minloss_val_biases+count, 1);
                }
                free(count_bias);
                free(val_bias);
            }
            if(1){
                get_topk_float(a->grads_val_biases[i], a->grads_loc_biases[i], iter*a->topk_bias, a->topk_biases[i], maxloss_val_biases + count, maxloss_loc_biases + count, 0);
                if(a->worst) get_topk_float(a->grads_val_biases_min[i], a->grads_loc_biases_min[i], iter*a->topk_bias, a->topk_biases[i], minloss_val_biases + count, minloss_loc_biases + count, 1);
            }
        }
    }
    
    /*printf("l*topk max value and loc of bias\n");
    for(int i = 0; i < layer_num; i++){
        for(int j = 0; j < a->topk_bias; j++){
            printf("%f ", maxloss_val_biases[i*a->topk_bias+j]);
        }
        printf("\n");
    }
    for(int i = 0; i < layer_num; i++){
        for(int j = 0; j < a->topk_weight; j++){
            printf("%d ", maxloss_loc_weights[i*a->topk_weight+j]);
        }
        printf("\n");
    }*/
    
    int offset_weight = a->topk_weight;
    int offset_bias = a->topk_bias;
    get_topk_with_layer(maxloss_val_weights, maxloss_loc_weights, a->topk_weight*layer_num, a->topk_weight, a->mloss_loc_weights[0], a->mloss_loc_weights[1], 0);
    get_topk_with_layer(maxloss_val_biases, maxloss_loc_biases, a->topk_bias*layer_num, a->topk_bias, a->mloss_loc_biases[0], a->mloss_loc_biases[1], 0);
    if(a->worst){
        get_topk_with_layer(minloss_val_weights, minloss_loc_weights, a->topk_weight*layer_num, a->topk_weight, a->mloss_loc_weights[0]+offset_weight, a->mloss_loc_weights[1]+offset_weight, 1);
        get_topk_with_layer(minloss_val_biases, minloss_loc_biases, a->topk_bias*layer_num, a->topk_bias, a->mloss_loc_biases[0]+offset_bias, a->mloss_loc_biases[1]+offset_bias, 1);
    }

   
    printf("topk max/min layer loc of bias\n");
    for(i = 0; i < a->topk_bias; i++) printf("%d ", a->mloss_loc_biases[0][i]);
    printf("\n");
    for(i = 0; i < a->topk_bias; i++) printf("%d ", a->mloss_loc_biases[1][i]);
    printf("\n");

    if(a->worst){
        for(i = 0; i < a->topk_bias; i++) printf("%d ", a->mloss_loc_biases[0][i+offset_bias]);
        printf("\n");
        for(i = 0; i < a->topk_bias; i++) printf("%d ", a->mloss_loc_biases[1][i+offset_bias]);
        printf("\n");
    }
    /*
    {
        int layer_idx, idx;
        FILE *fp = fopen("./grad_weight_bias.log", "w+");
        for(i = 0; i < a->topk_weight; i++){
            layer_idx = a->mloss_loc_weights[0][i];
            idx = a->mloss_loc_weights[1][i];
            fprintf(fp, "%f ", a->weights[layer_idx][idx]);
        }
        if(a->worst){
            for(i = a->topk_weight-1; i >= 0; i--){
                layer_idx = a->mloss_loc_weights[0][offset_weight+i];
                idx = a->mloss_loc_weights[1][offset_weight+i];
                fprintf(fp, "%f ", a->weights[layer_idx][idx]);
            }
        }
        fprintf(fp, "\n");
        for(i = 0; i < a->topk_weight; i++){
            fprintf(fp, "%f ", maxloss_val_weights[i]);
        }
        if(a->worst){
            for(i = a->topk_weight-1; i >= 0; i--){
                fprintf(fp, "%f ", minloss_val_weights[i]);
            }
        }
        fprintf(fp, "\n");
        for(i = 0; i < a->topk_bias; i++){
            layer_idx = a->mloss_loc_biases[0][i];
            idx = a->mloss_loc_biases[1][i];
            fprintf(fp, "%f ", a->biases[layer_idx][idx]);
        }
        if(a->worst){
            for(i = a->topk_bias-1; i >= 0; i--){
                layer_idx = a->mloss_loc_biases[0][offset_bias+i];
                idx = a->mloss_loc_biases[1][offset_bias+i];
                fprintf(fp, "%f ", a->biases[layer_idx][idx]);
            }
        }
        fprintf(fp, "\n");
        for(i = 0; i < a->topk_bias; i++){
            fprintf(fp, "%f ", maxloss_val_biases[i]);
        }
        for(i = a->topk_bias-1; i >= 0; i--){
            fprintf(fp, "%f ", minloss_val_biases[i]);
        }
        fprintf(fp, "\n");
        fclose(fp);
    }*/

    free(maxloss_val_weights);
    free(maxloss_val_biases);
    free(maxloss_loc_weights);
    free(maxloss_loc_biases);
    free(minloss_val_weights);
    free(minloss_val_biases);
    free(minloss_loc_weights);
    free(minloss_loc_biases);
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
        case 2:
            a->layer_num = layer_num;
            a->len = a->flt_len;
            a->topk = a->topk_flt;
            a->mloss = a->mloss_flt;
            a->macc = a->macc_flt;
            a->mloss_loc = a->mloss_loc_flt;
            a->avf = a->avf_flt;
            break;
         case 3:
            a->layer_num = layer_num;
            a->len = a->l_len;
            a->topk = a->topk_l;
            a->x = a->weights;
            a->x_gpu = a->weights_gpu;
            a->mloss = a->mloss_l;
            a->macc = a->macc_l;
            a->mloss_loc = a->mloss_loc_l;
            a->avf = a->avf_l;
            break;

        default:
            printf("wrong attack type!\n");
            return;
    }

    int topk = a->worst? 2*a->topk: a->topk;

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

        for(k = 0; k < topk; k++){
            a->k_idx = k;
            if(a->progress_attack){
                for(m = 0; m < a->fb_len; m++){
                    a->bit_idx = a->flipped_bit[m];
                    //printf("start attack\n");
                    a->single_attack(net);
                    //printf("\nend attack\n");
                }
                a->mloss[a->fb_len*k] += network_predict_attack(net, val) / iter;
                int nboxes = 0;
                detection *dets = get_network_boxes(net, net->w, net->h, thresh, .5, 0, 1, &nboxes);
                if (nms) do_nms_sort(dets, nboxes, classes, nms);

                a->macc[a->fb_len*k] += cal_map(net, dets, args.boxes, nboxes, args.num_boxes, iou_thresh, thresh_calc_avg_iou) / iter;
                free_detections(dets, nboxes);
                for(m = 0; m < a->fb_len; m++){
                    a->bit_idx = a->flipped_bit[m];
                    //printf("start attack\n");
                    a->single_attack(net);
                    //printf("\nend attack\n");
                }
                //printf("back!\n");
            }
            else{
                for(m = 0; m < a->fb_len; m++){
                    a->bit_idx = a->flipped_bit[m];
                    a->single_attack(net);
                    a->mloss[a->fb_len*k+m] += network_predict_attack(net, val) / iter;
                    int nboxes = 0;
                    detection *dets = get_network_boxes(net, net->w, net->h, thresh, .5, 0, 1, &nboxes); 
                    if (nms) do_nms_sort(dets, nboxes, classes, nms);

                    a->macc[a->fb_len*k+m] += cal_map(net, dets, args.boxes, nboxes, args.num_boxes, iou_thresh, thresh_calc_avg_iou) / iter;
                    free_detections(dets, nboxes);

                    a->single_attack(net);
                    //fprintf(stderr, "attack time: %f\n", what_time_is_it_now() - time);
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
        args.num_boxes = count_boxes;

        load_thread = load_data_in_thread(args);
        free_data(val);
        a->seen_img += args.n;
    }

    a->loss_thresh = avg_loss;
    a->acc_thresh = avg_acc;
    printf("loss thresh: %f, acc thresh: %f\n", avg_loss, avg_acc);
    printf("type: %d\n", type);
    fprintf(avf_fp, "type: %d\n", type);
    cal_avf(a, avf_fp);
    {
        for(i = 0; i < topk; i++) printf("%f ", a->avf[i]);
        printf("\n");
    }
    print_avf_log(a, avf_fp);
    if(type == 0) a->a_weight = 0;
    if(type == 1) a->a_bias = 0;
    a->seen_img = 0;
}

void cal_avf(attack_args *a, FILE *fp)
{
    if(a->avf_type == 0) cal_avf_0(a, fp);  //数个数法, 归一化
    else if(a->avf_type == 1) cal_avf_1(a, fp);  //各个bit取最大最小，归一
    else cal_avf_2(a, fp);  //取平均值，未归一
}

void cal_avf_0(attack_args *a, FILE *fp)
{
    int i, j, m;
    int topk = a->worst? 2*a->topk: a->topk;
    int fb_len = a->fb_len;
    float loss_thresh = a->loss_thresh;
    float acc_thresh = a->acc_thresh;

    float *losses = (float *)calloc(topk, sizeof(float));
    float *accs = (float *)calloc(topk, sizeof(float));

    float *avf = a->avf;
    float *mloss = a->mloss;
    float *macc = a->macc;

    float t_loss;
    float t_acc;
    printf("mloss and macc: \n");
    for(i = 0; i < topk*fb_len; i++) printf("%f ", mloss[i]);
    printf("\n");
    for(i = 0; i < topk*fb_len; i++) printf("%f ", macc[i]);
    printf("\n");

    for(j = 0; j < topk; j++){
        for(m = 0; m < fb_len; m++){
            t_loss = mloss[j*fb_len+m];
            t_acc = macc[j*fb_len+m];
            if(isnan(t_loss) || isinf(t_loss) || t_loss > loss_thresh) losses[j]++;
            if(t_acc >= acc_thresh) accs[j]++;
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
}

void cal_avf_1(attack_args *a, FILE *fp)
{
    int i, j, m;
    int topk = a->worst? 2*a->topk: a->topk;
    int fb_len = a->fb_len;
    //float loss_thresh = a->loss_thresh;
    //float acc_thresh = a->acc_thresh;

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

            if(max_acc[m] == 0) macc[j*fb_len+m] = 0;
            else if(min_acc[m] == max_acc[m]) macc[j*fb_len+m] = 0.5;
            else macc[j*fb_len+m] = (macc[j*fb_len+m] - min_acc[m]) / (max_acc[m] - min_acc[m]); 
            losses[j] += mloss[j*fb_len+m] / fb_len;
            accs[j] += macc[j*fb_len+m] / fb_len;
        }
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


void cal_avf_2(attack_args *a, FILE *fp)
{
    int i, j;
    int topk = a->worst? 2*a->topk: a->topk;
    int fb_len = a->fb_len;
    //float loss_thresh = a->loss_thresh;
    //float acc_thresh = a->acc_thresh;

    float *losses = (float *)calloc(topk, sizeof(float));
    float *accs = (float *)calloc(topk, sizeof(float));

    float *avf = a->avf;   //topk
    float *mloss = a->mloss;  //(topk) * fb_len
    float *macc = a->macc;

    for(i = 0; i < topk; i++){
        for(j = 0; j < fb_len; j++){
            losses[i] += mloss[i*fb_len+j] / fb_len;
            accs[i] += macc[i*fb_len+j] / fb_len;
        }
        avf[i] = losses[i] * a->alpha + (1 - accs[i]) * (1 - a->alpha) ;
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
    int topk = a->worst? 2*a->topk: a->topk;
    fprintf(fp, "avf: \n");
    for(i = 0; i < topk; i++)
        fprintf(fp, "%f ", a->avf[i]);
    fprintf(fp, "\n");
}
