#ifndef SORT_H
#define SORT_H

void qsort_topk_int_float(int *a, float *b, int *c, int l, int r, int k, int reverse);
void qsort_topk_float_int(float *a, int *b, int l, int r, int k, int reverse);
void qsort_with_layer(float *a, int *b, int *c, int l, int r, int reverse);

void heapsort_topk_int_float(int *a, float *b, int *c, int n, int k, int reverse, int *idx, float *val);
void heapsort_topk_float_int(float *a, int *b, int n, int k, int reverse, int *idx, float *val);
void heapsort_topk_with_layer(float *a, int *b, int *c, int n, int k, int reverse, int *lidx, int *idx);



#endif
