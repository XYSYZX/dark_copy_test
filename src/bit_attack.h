#ifndef BIT_ATTACK_H
#define BIT_ATTACK_H

#include "darknet.h"

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

float **make_2d_array_float(int n, int m);
int **make_2d_array_int(int n, int m);
void free_2d_array_float(float **x, int n);
void free_2d_array_int(int **x, int n);
void zero_2d_array_float(float **x, int n, int m);
void zero_2d_array_int(int **x, int n, int m);
void print_2d_array_int(int **x, int m, int n);
void print_2d_array_float(float **x, int m, int n);

void free_attack_args(attack_args a);
void set_attack_args(network *net);
float sign(float x);
void sign_attacker(float *x, int *loc, int topk, float *grad, float epsilon);
void sign_delete(float *x, int *loc, int topk, float *grad, float epsilon);
void bit_flip_attacker(attack_args a);
//void bit_flip_delete(attack_args a);
void get_max_loss(attack_args *attack);
void get_avf(network *net, load_args args, int type);
void cal_avf(attack_args *a);
void get_topk_grad(attack_args *a);
float cal_map(network *net, detection *dets, box_label *truth, int nboxes, int num_labels, float iou_thresh, float thresh_calc_avg_iou);
#ifdef GPU
/*
#ifdef __cplusplus
extern "C" {
#endif
*/
void sign_delete_gpu(float *x_gpu, int *loc, int topk, float *grad_gpu, float epsilon);
void sign_attacker_gpu(float *x_gpu, int *loc, int topk, float *grad_gpu, float epsilon);
//void bit_flip_delete_gpu(attack_args a, float *x_gpu, int *loc, int topk, float *grad_gpu);
void bit_flip_attacker_gpu(attack_args a);
/*
#ifdef __cpluscplus
}
#endif
*/

#endif
#endif
