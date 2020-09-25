#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234
extern int gpu_index;

//all struct in darknet.h:  metadata, tree, update_args, network, layer, augment_args, image, box, detection, matrix, data, load_args, box_label, node, list
//all enum in darknet.h: ACTIVATION, IMTYPE, BINARY_ACTIVATION, LAYER_TYPE, COST_TYPE, learning_rate_policy, data_type
typedef struct{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);   //option_list.c

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;
tree *read_tree(char *filename);   //in tree.c

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;

typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct attack_args;
typedef struct attack_args attack_args;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    void (*forward_gpu)   (struct layer, struct network);
    void (*backward_gpu)  (struct layer, struct network);
    void (*update_gpu)    (struct layer, update_args);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;   //输入图像高宽通道
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;
    
    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu; 

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
	
    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;

    float * max_weight;
    float * min_weight;
    float * max_output;
    float * min_output;

#ifdef GPU
    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;

    float *temp_gpu;
    float *temp2_gpu;
    float *temp3_gpu;

    float *dh_gpu;
    float *hh_gpu;
    float *prev_cell_gpu;
    float *cell_gpu;
    float *f_gpu;
    float *i_gpu;
    float *g_gpu;
    float *o_gpu;
    float *c_gpu;
    float *dc_gpu; 

    float *m_gpu;
    float *v_gpu;
    float *bias_m_gpu;
    float *scale_m_gpu;
    float *bias_v_gpu;
    float *scale_v_gpu;

    float * combine_gpu;
    float * combine_delta_gpu;

    float * prev_state_gpu;
    float * forgot_state_gpu;
    float * forgot_delta_gpu;
    float * state_gpu;
    float * state_delta_gpu;
    float * gate_gpu;
    float * gate_delta_gpu;
    float * save_gpu;
    float * save_delta_gpu;
    float * concat_gpu;
    float * concat_delta_gpu;

    float * binary_input_gpu;
    float * binary_weights_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;
    float * weight_change_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;
    float * bias_change_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;
    float * scale_change_gpu;

    float * output_gpu;
    float * loss_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;

#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

void free_layer(layer);

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    int n;              //网络的层数，调用make_network(int n)时赋值
    int batch;          //一批训练中的图片参数，和subdivsions参数相关
    size_t *seen;       //目前已经读入的图片张数(网络已经处理的图片张数)
    int *t;
    float epoch;
    int subdivisions;   //batch/subdivision is the num of samples sending into trainer at one time, which means send a batch of sample by (subdivision) times
    layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;   // 中间变量，与上面的 input 对应，用来暂存 input 数据对应的标签数据（真实数）
    float *delta;   // 中间变量，用来暂存某层网络的敏感度图（反向传播处理当前层时，用来存储上一层的敏感度图，因为当前层会计算部分上一层的敏感度图，可以参看 network.c 中的 backward_network() 函数）
    float *workspace; // 网络的工作空间, 指的是所有层中占用运算空间最大的那个层的 workspace_size
    int train;  // 网络是否处于训练阶段的标志参数，如果是则值为1.
    int index;  // 标志参数，当前网络的活跃层 
    float *cost;
    float clip;

#ifdef GPU
    float *input_gpu;
    float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;
#endif
//start dirty!!!
    int dirty;
    int dirty_limit;
    int robust;
//end dirty
    int output_bound;
    int weight_bound;
    int limit_weight;
    int limit_output;
//bit attack
    int bit_attack;
    attack_args *attack;

} network;

typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

typedef struct {
    int w;
    int h;
    int c;  //channel of color, rgb..
    float *data;
} image;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;  //probability of classes
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct{  //record box and probs
    int box_id;
    float x, y, w, h;
    float prob[80];
} boxing;

//from here, funcs are written by zyc
typedef struct gold_ans{  //record a golden output
    box bbox;
    int img_id;
    int classes;
    int box_num;
    float objectness;
    float *prob;
} gold_ans;

typedef struct fileline{
        char *line_name;
        int box_num;
        boxing *boxes;
        int **classes; //a box might have man labels, this is many labels for many boxes
}fileline;
//stop here

typedef struct matrix{
    int rows, cols;  //rows是一次加载到内存中的样本的个数（batch*net.subdivisions），cols就是样本的维度，**vals指向的是样本的值
    float **vals;
} matrix;



typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA, ATTACK_DATA
} data_type;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
    box_label *boxes;
    //int num_labels;
} load_args;

network *load_network(char *cfg, char *weights, int clear); // in network.c
load_args get_base_args(network *net);                      // in network.c
void get_weight_bound_network(network *net);
void get_output_bound_network(network *net);
void inject_noise_weights(network *netp);
void inject_noise_weights_limit(network *netp);
void inject_noise_weights_onebit(layer *l, int weight_idx, int bit_idx);

void free_data(data d);                                     //in data.c

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

pthread_t load_data(load_args args);
list *read_data_cfg(char *filename);   //option_list.c
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);   //utils.c
data resize_data(data orig, int w, int h);
data *tile_data(data orig, int divs, int size);
data select_data(data *orig, int *inds);

void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);


float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax(float *input, int n, float temp, int stride, float *output);
void abs_cpu(float *X, float *Y, size_t N);

int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void fill_gpu(int N, float ALPHA, float * X, int INCX);  //in blas_kernel.cu
void scal_gpu(int N, float ALPHA, float * X, int INCX);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
void abs_gpu(float *X, float *Y, size_t N);

void cuda_set_device(int n);
void cuda_free(float *x_gpu);
void cuda_free_int(int *x_gpu);
float *cuda_make_array(float *x, size_t n);
float *cuda_make_array_dev(float *x_gpu, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
void cuda_push_array_dev(float *y_gpu, float *x_gpu, size_t n);
void cuda_pull_int_array(int *x_gpu, int *x, size_t n);
float cuda_mag_array(float *x_gpu, size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);

void forward_network_gpu(network *net);     //network.c
void backward_network_gpu(network *net);    //network.c
void update_network_gpu(network *net);      //same
void inject_noise_weights_gpu(network *netp);
void inject_noise_weights_limit_gpu(network *netp);
void inject_noise_weights_onebit_gpu(layer *l, int weight_idx, int bit_idx);

float train_networks(network **nets, int n, data d, int interval);   //network.c
void sync_nets(network **nets, int n, int interval);                 //network.c
void harmless_update_network_gpu(network *net);                      //network.c
#endif
image get_label(image **characters, char *string, int size);
void draw_label(image a, int r, int c, image label, const float *rgb);
void save_image(image im, const char *name);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void get_data_single(data d, float *X, float *y);
void grayscale_image_3c(image im);
void normalize_image(image p);
void matrix_to_csv(matrix m);
float train_network_sgd(network *net, data d, int n);    //network
void rgbgr_image(image im);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_cifar10_data(char *filename);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix csv_to_matrix(char *filename);
float *network_accuracies(network *net, data d, int n);
float train_network_datum(network *net);                 //network
image make_random_image(int w, int h, int c);

void denormalize_connected_layer(layer l);
void denormalize_convolutional_layer(layer l);
void statistics_connected_layer(layer l);
void rescale_weights(layer l, float scale, float trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen);
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

char *option_find_str(list *l, char *key, char *def);   //option_list.c
int option_find_int(list *l, char *key, int def);      //option_list.c
int option_find_int_quiet(list *l, char *key, int def);  //option_list.c
void free_list_contents_kvp(list *l);

network *parse_network_cfg(char *filename);          //parser.c
void save_weights(network *net, char *filename);     //same
void load_weights(network *net, char *filename);  
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);  //parser.c

//something add by myself
void get_weight_bound(layer l);                     //bound_adjust
void get_output_bound(layer l);
//void constrain(float *values, int max, int min, int nums);
void save_weight_bound(network *net, char *file);
void save_output_bound(network *net, char *file);
void load_weight_bound(network *net, char *file);
void load_output_bound(network *net, char *file);
void load_output_bound_2(network *net, char *file);
void load_output_bound_8(network *net, char *file);
void check_weights(network *net);
void check_outputs(layer l);

void inject_noise_float(float *w, unsigned int length);  //noise_inject.c
void inject_noise_float_limit(float *w, unsigned int length, int *limit, int limits);
void inject_noise_float_onebit(float *w, int idx, int bit_idx);
void inject_noise_float_manybit(float *w, int idx, int bit_len, int *bit_idxs);
#ifdef GPU
void get_output_bound_gpu(layer l);
void get_weight_bound_gpu(layer l);
//void constrain_gpu(float *values, int max, int min, int nums);
void check_weights_gpu(network *net);
void check_outputs_gpu(layer l);

void inject_noise_float_gpu(float *w, unsigned int length);  //noise_inject_kernel
void inject_noise_float_limit_gpu(float *w, unsigned int length, int *limit, int limits);
void inject_noise_float_onebit_gpu(float *w_gpu, int idx, int bit_idx);
void inject_noise_float_manybit_gpu(float *gpu, int idx, int bit_len, int *bit_idxs);
void test_inject_noise_gpu();
#endif

void zero_objectness(layer l);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
void free_network(network *net);
void set_batch_network(network *net, int b);
void set_temp_network(network *net, float t);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void censor_image(image im, int dx, int dy, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, float thresh);
image mask_to_rgb(image mask);
int resize_network(network *net, int w, int h);
void free_matrix(matrix m);
void test_resize(char *filename);
int show_image(image p, const char *name, int ms);
image copy_image(image p);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
float get_current_rate(network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
size_t get_current_batch(network *net);
void constrain_image(image im);
image get_network_image_layer(network *net, int i);
gold_ans *create_network_boxes(int img_id, int box_num, int class_num);
layer get_network_output_layer(network *net);
void top_predictions(network *net, int n, int *index);
void get_topk(float *x, float *x_gpu, int length, int *w_idx, int topk);
void flip_image(image a);
image float_to_image(int w, int h, int c, float *data);
void ghost_image(image source, image dest, int dx, int dy);
float network_accuracy(network *net, data d);
void random_distort_image(image im, float hue, float saturation, float exposure);
void fill_image(image m, float s);
image grayscale_image(image im);
void rotate_image_cw(image im, int times);
double what_time_is_it_now();
void save_cost(int layer_idx, int weight_idx, int bit_idx, float *cost, FILE *f);
image rotate_image(image m, float rad);
void visualize_network(network *net);
float box_iou(box a, box b);
float box_area_df(box a, box b);
int inclusion(box a, box b, float max);
data load_all_cifar10();
box_label *read_boxes(char *filename, int *n);
box float_to_box(float *f, int stride);
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);
void draw_detections_mul(char *imagename, char *odir, image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);
int compare_detections(int img_num, detection *dets, int box_num, int classes, FILE *wf, FILE *fp, float df, gold_ans *gd);
//char *write_detections(char *rightline, int img_id, detection *dets, int num, float thresh, int classes, int max_box);
matrix network_predict_data(network *net, data test);
//gold_ans *parse_goldans(char *rightline, int max_num);
void save_rightline(int img_id, detection *dets, int nboxes, int classes, int max_box, FILE *rf);
gold_ans *load_rightline(FILE *rf);
//void save_target(FILE *f, float *target, int img_id, int outputs);
//float *load_target(FILE *f, int img_id_t);

image **load_alphabet();
image get_network_image(network *net);
float *network_predict(network *net, float *input);
float *network_predict_single(network *net, data d);
void set_network_input_truth(network *net, data d);
float network_predict_search(network *net, data d);
float network_predict_attack(network *net, data d);

int network_width(network *net);
int network_height(network *net);
float *network_predict_image(network *net, image im);
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);
void free_gold_ans(gold_ans *gd);

void reset_network_state(network *net, int b);

char **get_labels(char *filename);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh);

matrix make_matrix(int rows, int cols);

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
void make_window(char *name, int w, int h, int fullscreen);
#endif

void free_image(image m);
float train_network(network *net, data d);
pthread_t load_data_in_thread(load_args args);
void load_data_blocking(load_args args);
list *get_paths(char *filename);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
float sec(clock_t clocks);
void **list_to_array(list *l);
void top_k(float *a, int n, int k, int *index);
void top_k_int(int *a, int n, int k, int *index, int *y);
int *read_map(char *filename);
void error(const char *s);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
int *random_index_order(int min, int max);
void free_list(list *l);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
void scale_array(float *a, int n, float s);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
float rand_normal();
float rand_uniform(float min, float max);

typedef struct attack_args{
    //network *net;
    int layer_num;
    int iter;
    int layer_idx;
    int iter_idx;
    int k_idx;
    int bit_idx;

    int sign_attack;  //FGSM攻击
    //int progress_attack; //一次攻击好几位
    float epsilon;    //FGSM攻击因数
    int reverse;      //撤除攻击

    float alpha;  //控制loss和accuracy占avf比例
    float loss_thresh;  //判断loss的阈值
    float acc_thresh;   //判断accurary的阈值

    int a_input;   //attack inputs, 0 or 1
    int a_weight;
    int a_bias;
    int a_output;

    int topk;
    int topk_inputs; //num of inputs to be attacked
    int topk_weights;
    int topk_biases;
    int topk_outputs;
    int *topks;
    
    int *flipped_bit;   //一个数字中要翻转的位数
    int fb_len;        //位数的长度

    int total_img;
    int seen_img;
    int n;  //一次看几张图片

    int **grads_loc;
    int **grads_loc_inputs;    //length: 1 * (topk * iter)
    int **grads_loc_weights;   //length: layers * (topk * iter)
    int **grads_loc_biases;
    int **grads_loc_outputs;

    int **mloss_loc;
    int **mloss_loc_inputs;   //长度: 1 * (topk)
    int **mloss_loc_weights; //layer1, weight1, weight2... 长度: layers * (topk)
    int **mloss_loc_outputs; //layer1, output1, output2..., 长度同上
    int **mloss_loc_biases;

    float **mloss;
    float **mloss_inputs; //长度: 1 * (topk * fb_len)
    float **mloss_weights; //长度: layers * (topk * fb_len)
    float **mloss_biases;
    float **mloss_outputs; //长度同上

    float **macc;
    float **macc_inputs; //长度: 1 * (topk * fb_len)
    float **macc_weights; //长度: layers * (topk * fb_len)
    float **macc_biases;
    float **macc_outputs; //长度同上

    float **avf;
    float **avf_inputs; //长度: 1 * (topk)
    float **avf_weights; //长度: layers * (topk)
    float **avf_biases;
    float **avf_outputs; //长度同上

    int *len;
    int *inputs_len;    //length: 1
    int *weights_len;   //length: layer num
    int *biases_len;
    int *outputs_len;

    float **grads;
    float **grads_gpu;
    float **input_grads;    //length: 1*1
    float **input_grads_gpu;
    float **weight_grads;   //length: layer num*1
    float **weight_grads_gpu;
    float **bias_grads;
    float **bias_grads_gpu;
    float **output_grads;
    float **output_grads_gpu;

    float **x;
    float **x_gpu;
    float **inputs;     //length: 1*1
    float **inputs_gpu;
    float **weights;    //length: layer num*1
    float **weights_gpu;
    float **biases;
    float **biases_gpu;
    float **outputs;
    float **outputs_gpu;
} attack_args;

void attack_data(network *net, load_args args, load_args val_args);
//void progressive_attack(network *net);
void single_attack(network *net);
#ifdef __cplusplus
}
#endif
#endif
