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
void sign_attacker(float *x, float *grad, int idx, float epsilon);
void sign_delete(float *x, float *grad, int idx, float epsilon);
void bit_flip_attacker(float *x, int idx, int bit_idx);

void cal_grad_flt(attack_args *a, float *flt_grad);
void get_topk_grad_wb(attack_args *a);
void get_topk_grad_flt(attack_args *a);
void get_topk_grad_l(attack_args *a);
void get_max_loss_wb(attack_args *attack);
void get_max_loss_flt(attack_args *attack);
void get_max_loss_l(attack_args *a);
void get_avf(network *net, load_args args, int type, FILE *avf_fp);
void single_attack_wb(network *net);
void single_attack_flt(network *net);
void single_attack_l(network *net);

void cal_avf(attack_args *a, FILE *fp);
void cal_avf_0(attack_args *a, FILE *fp);
void cal_avf_1(attack_args *a, FILE *fp);
void cal_avf_2(attack_args *a, FILE *fp);
float cal_map(network *net, detection *dets, box_label *truth, int nboxes, int num_labels, float iou_thresh, float thresh_calc_avg_iou);
void print_avf_log(attack_args *a, FILE *fp);

#ifdef GPU
void sign_delete_gpu(float *x_gpu, float *grad_gpu, int idx, float epsilon);
void sign_attacker_gpu(float *x_gpu, float *grad_gpu, int idx, float epsilon);
void bit_flip_attacker_gpu(float *x_gpu, int idx, int bit_idx);
#endif

#endif
