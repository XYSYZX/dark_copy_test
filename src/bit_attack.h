#ifndef BIT_ATTACK_H
#define BIT_ATTACK_H

#include "darknet.h"

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
void get_max_loss(attack_args *attack);
void get_avf(network *net, load_args args, int type);
void cal_avf(attack_args *a, float avg_loss);
void get_topk_grad(attack_args *a);
#ifdef GPU
void sign_delete_gpu(float *x_gpu, int *loc, int topk, float *grad_gpu, float epsilon);
void sign_attacker_gpu(float *x_gpu, int *loc, int topk, float *grad_gpu, float epsilon);
#endif
#endif
