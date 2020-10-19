#include <stdlib.h>
//#include "darknet.h"
#include "sort.h"


//qsort series
int partition_int_idx(int *a, float *b, int *c, int l, int r)
{
    int x = a[l], z = c[l];
    int m = l, n = r;
    float y = b[l];
    while(1){
        while(n > m && a[n] <= x) n--;
        while(m < n && a[m] >= x) m++;
        if(m >= n) break;
        int tmp_a = a[n];
        float tmp_b = b[n];
        int tmp_c = c[n];
        a[n] = a[m];
        b[n] = b[m];
        c[n] = c[m];
        a[m] = tmp_a;
        b[m] = tmp_b;
        c[m] = tmp_c;
    }
    a[l] = a[n];
    b[l] = b[n];
    c[l] = c[n];
    a[n] = x;
    b[n] = y;
    c[n] = z;
    return n;
}

int partition_int_idx_reverse(int *a, float *b, int *c, int l, int r)
{
    int x = a[l], z = c[l];
    int m = l, n = r;
    float y = b[l];
    while(1){
        while(n > m && a[n] >= x) n--;
        while(m < n && a[m] <= x) m++;
        if(m >= n) break;
        int tmp_a = a[n];
        float tmp_b = b[n];
        int tmp_c = c[n];
        a[n] = a[m];
        b[n] = b[m];
        c[n] = c[m];
        a[m] = tmp_a;
        b[m] = tmp_b;
        c[m] = tmp_c;
    }
    a[l] = a[n];
    b[l] = b[n];
    c[l] = c[n];
    a[n] = x;
    b[n] = y;
    c[n] = z;
    return n;
}

void qsort_topk_int_float(int *a, float *b, int *c, int l, int r, int k, int reverse)
{
    int j = reverse? partition_int_idx_reverse(a, b, c, l ,r): partition_int_idx(a, b, c, l ,r);
    if(j == k){
        return;
    }
    return j > k? qsort_topk_int_float(a, b, c, l, j-1, k, reverse): qsort_topk_int_float(a, b, c, j+1, r, k, reverse);
}

int partition_float_idx(float *a, int *b, int l, int r)
{
    float x = a[l];
    int m = l, n = r, idx = b[l];
    while(1){
        while(n > m && a[n] <= x) n--;
        while(m < n && a[m] >= x) m++;
        if(m >= n) break;
        float tmp = a[n];
        int tmp_idx = b[n];
        a[n] = a[m];
        b[n] = b[m];
        a[m] = tmp;
        b[m] = tmp_idx;
    }
    a[l] = a[n];
    b[l] = b[n];
    a[n] = x;
    b[n] = idx;
    return n;
}

int partition_float_idx_reverse(float *a, int *b, int l, int r)
{
    float x = a[l];
    int m = l, n = r, idx = b[l];
    while(1){
        while(n > m && a[n] >= x) n--;
        while(m < n && a[m] <= x) m++;
        if(m >= n) break;
        float tmp = a[n];
        int tmp_idx = b[n];
        a[n] = a[m];
        b[n] = b[m];
        a[m] = tmp;
        b[m] = tmp_idx;
    }
    a[l] = a[n];
    b[l] = b[n];
    a[n] = x;
    b[n] = idx;
    return n;
}

void qsort_topk_float_int(float *a, int *b, int l, int r, int k, int reverse)
{
    int j = reverse? partition_float_idx_reverse(a, b, l ,r): partition_float_idx(a, b, l ,r);
    if(j == k){
        return;
    }
    return j > k? qsort_topk_float_int(a, b, l, j-1, k, reverse): qsort_topk_float_int(a, b, j+1, r, k, reverse);
}

int partition_float_with_layer(float *a, int *b, int *layer_loc, int l, int r)
{
    float x = a[l];
    int m = l, n = r, idx = b[l], layer_idx = layer_loc[l];
    while(1){
        while(n > m && a[n] <= x) n--;
        while(m < n && a[m] >= x) m++;
        if(m >= n) break;
        float tmp = a[n];
        int tmp_idx = b[n];
        int tmp_layer = layer_loc[n];
        a[n] = a[m];
        b[n] = b[m];
        layer_loc[n] = layer_loc[m];
        a[m] = tmp;
        b[m] = tmp_idx;
        layer_loc[m] = tmp_layer;
    }
    a[l] = a[n];
    b[l] = b[n];
    layer_loc[l] = layer_loc[n];
    a[n] = x;
    b[n] = idx;
    layer_loc[n] = layer_idx;
    return n;
}

int partition_float_with_layer_reverse(float *a, int *b, int *layer_loc, int l, int r)
{
    float x = a[l];
    int m = l, n = r, idx = b[l], layer_idx = layer_loc[l];
    while(1){
        while(n > m && a[n] >= x) n--;
        while(m < n && a[m] <= x) m++;
        if(m >= n) break;
        float tmp = a[n];
        int tmp_idx = b[n];
        int tmp_layer = layer_loc[n];
        a[n] = a[m];
        b[n] = b[m];
        layer_loc[n] = layer_loc[m];
        a[m] = tmp;
        b[m] = tmp_idx;
        layer_loc[m] = tmp_layer;
    }
    a[l] = a[n];
    b[l] = b[n];
    layer_loc[l] = layer_loc[n];
    a[n] = x;
    b[n] = idx;
    layer_loc[n] = layer_idx;
    return n;
}

void qsort_with_layer(float *a, int *b, int *c, int l, int r, int reverse)
{
    if(l >= r) return;
    int j = reverse? partition_float_with_layer_reverse(a, b, c, l ,r): partition_float_with_layer(a, b, c, l ,r);
    qsort_with_layer(a, b, c, l, j-1, reverse);
    qsort_with_layer(a, b, c, j+1, r, reverse);
}


//heap sort series
void down_topk_int_float(int *a, float *b, int *c, int n, int k)  //从大到小
{
    int root = n;  //根结点索引
    int lchild = 2*root + 1; //根结点的左孩子
    int child = lchild;

    int parent = a[root];
    float parent_b = b[root];
    int parent_c = c[root];

    //下沉
    while(lchild < k){
        if(lchild+1 < k && a[lchild+1] > a[lchild])
            child = lchild + 1;
        if(parent < a[child]){
            a[root] = a[child];
            b[root] = b[child];
            c[root] = c[child];
            root = child;
            lchild = 2*root + 1;
            child = lchild;
        }
        else break;
    }
    a[root] = parent;
    b[root] = parent_b;
    c[root] = parent_c;
}

void up_topk_int_float(int *a, float *b, int *c, int n, int k)
{
    int root = n;  //根结点索引
    int lchild = 2*root + 1; //根结点的左孩子
    int child = lchild;

    int parent = a[root];
    float parent_b = b[root];
    int parent_c = c[root];

    //下沉
    while(lchild < k){
        if(lchild+1 < k && a[lchild+1] < a[lchild])
            child = lchild + 1;
        if(parent > a[child]){
            a[root] = a[child];
            b[root] = b[child];
            c[root] = c[child];
            root = child;
            lchild = 2*root + 1;
            child = lchild;
        }
        else break;
    }
    a[root] = parent;
    b[root] = parent_b;
    c[root] = parent_c;
}


void heapsort_topk_int_float(int *a, float *b, int *c, int n, int k, int reverse, int *idx, float *val)
{
    int i;
    int *res = (int *)calloc(k, sizeof(int));
    for(i = 0; i < k; i++){
        res[i] = a[i];
        val[i] = b[i];
        idx[i] = c[i];
    }
    for(i = (k>>1)-1; i>=0; i--) 
        reverse? down_topk_int_float(res, val, idx, i, k): up_topk_int_float(res, val, idx, i, k); /// 所有非叶子节点从后往前下沉
    for(i = k; i < n; i++){
        if(reverse && a[i] >= res[0]) continue;
        else if(!reverse && a[i] <= res[0]) continue;
        else{
            res[0] = a[i];
            val[0] = b[i];
            idx[0] = c[i];
            reverse? down_topk_int_float(res, val, idx, 0, k): up_topk_int_float(res, val, idx, 0, k);
        }
    }
    free(res);
}

void down_topk_float_int(float *a, int *b, int n, int k)  //从大到小
{
    int root = n;  //根结点索引
    int lchild = 2*root + 1; //根结点的左孩子
    int child = lchild;

    float parent = a[root];
    int parent_b = b[root];

    //下沉
    while(lchild < k){
        if(lchild+1 < k && a[lchild+1] > a[lchild])
            child = lchild + 1;
        if(parent < a[child]){
            a[root] = a[child];
            b[root] = b[child];
            root = child;
            lchild = 2*root + 1;
            child = lchild;
        }
        else break;
    }
    a[root] = parent;
    b[root] = parent_b;
}

void up_topk_float_int(float *a, int *b, int n, int k)
{
    int root = n;  //根结点索引
    int lchild = 2*root + 1; //根结点的左孩子
    int child = lchild;

    float parent = a[root];
    int parent_b = b[root];

    //下沉
    while(lchild < k){
        if(lchild+1 < k && a[lchild+1] < a[lchild])
            child = lchild + 1;
        if(parent > a[child]){
            a[root] = a[child];
            b[root] = b[child];
            root = child;
            lchild = 2*root + 1;
            child = lchild;
        }
        else break;
    }
    a[root] = parent;
    b[root] = parent_b;
}

void heapsort_topk_float_int(float *a, int *b, int n, int k, int reverse, int *idx, float *val)
{
    int i;
    for(i = 0; i < k; i++){
        val[i] = a[i];
        idx[i] = b[i];
    }
    for(i = (k>>1)-1; i>=0; i--) 
        reverse? down_topk_float_int(val, idx, i, k): up_topk_float_int(val, idx, i, k); /// 所有非叶子节点从后往前下沉
    for(i = k; i < n; i++){
        if(reverse && a[i] >= val[0]) continue;
        else if(!reverse && a[i] <= val[0]) continue;
        else{
            val[0] = a[i];
            idx[0] = b[i];
            reverse? down_topk_float_int(val, idx, 0, k): up_topk_float_int(val, idx, 0, k);
        }
    }
}

void down_topk_with_layer(float *a, int *b, int *c, int n, int k)  //从大到小
{
    int root = n;  //根结点索引
    int lchild = 2*root + 1; //根结点的左孩子
    int child = lchild;

    float parent = a[root];
    int parent_b = b[root];
    int parent_c = c[root];

    //下沉
    while(lchild < k){
        if(lchild+1 < k && a[lchild+1] > a[lchild])
            child = lchild + 1;
        if(parent < a[child]){
            a[root] = a[child];
            b[root] = b[child];
            c[root] = c[child];
            root = child;
            lchild = 2*root + 1;
            child = lchild;
        }
        else break;
    }
    a[root] = parent;
    b[root] = parent_b;
    c[root] = parent_c;
}

void up_topk_with_layer(float *a, int *b, int *c, int n, int k)
{
    int root = n;  //根结点索引
    int lchild = 2*root + 1; //根结点的左孩子
    int child = lchild;

    float parent = a[root];
    int parent_b = b[root];
    int parent_c = c[root];

    //下沉
    while(lchild < k){
        if(lchild+1 < k && a[lchild+1] < a[lchild])
            child = lchild + 1;
        if(parent > a[child]){
            a[root] = a[child];
            b[root] = b[child];
            c[root] = c[child];
            root = child;
            lchild = 2*root + 1;
            child = lchild;
        }
        else break;
    }
    a[root] = parent;
    b[root] = parent_b;
    c[root] = parent_c;
}

void heapsort_topk_with_layer(float *a, int *b, int *c, int n, int k, int reverse, int *lidx, int *idx)
{
    int i;
    float *res = (float *)calloc(k, sizeof(float));
    for(i = 0; i < k; i++){
        res[i] = a[i];
        lidx[i] = c[i];
        idx[i] = b[i];
    }
    for(i = (k>>1)-1; i>=0; i--)
        reverse? down_topk_with_layer(res, idx, lidx, i, k): up_topk_with_layer(res, idx, lidx, i, k);
    for(i = k; i < n; i++){
        if(reverse && a[i] >= res[0]) continue;
        else if(!reverse && a[i] <= res[0]) continue;
        else{
            res[0] = a[i];
            lidx[0] = c[i];
            idx[0] = b[i];
            reverse? down_topk_with_layer(res, idx, lidx, 0, k): up_topk_with_layer(res, idx, lidx, 0, k);
        }
    }
    printf("max/min layer value\n");
    for(i = 0; i < k; i++) printf("%f ", res[i]);
    printf("\n");
    free(res);
}

