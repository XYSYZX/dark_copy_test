#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
#include "darknet.h"
#include "list.h"

#define TIME(a) \
    do { \
    double start = what_time_is_it_now(); \
    a; \
    printf("%s took: %f seconds\n", #a, what_time_is_it_now() - start); \
    } while (0)

#define TWO_PI 6.2831853071795864769252866f

double what_time_is_it_now();
void shuffle(void *arr, size_t n, size_t size);
void sorta_shuffle(void *arr, size_t n, size_t size, size_t sections);
void free_ptrs(void **ptrs, int n);
int alphanum_to_int(char c);
char int_to_alphanum(int i);
int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);
int read_all_fail(int fd, char *buffer, size_t bytes);
int write_all_fail(int fd, char *buffer, size_t bytes);
void find_replace(char *str, char *orig, char *rep, char *output);
void malloc_error();
void file_error(char *s);
void strip(char *s);
void strip_char(char *s, char bad);
list *split_str(char *s, char delim);
char *fgetl(FILE *fp);
list *parse_csv_line(char *line);
char *copy_string(char *s);
int count_fields(char *line);
float *parse_fields(char *line, int n);
void translate_array(float *a, int n, float s);
float constrain(float min, float max, float a);
int constrain_int(int a, int min, int max);
float rand_scale(float s);
int rand_int(int min, int max);
void mean_arrays(float **a, int n, int els, float *avg);
float dist_array(float *a, float *b, int n, int sub);
float **one_hot_encode(float *a, int n, int k);
float sec(clock_t clocks);
void print_statistics(float *a, int n);
int int_index(int *a, int val, int n);

char *itoa(char *a, int n);
int par_atoi(char *a);
float par_atof(char *a);
char *copy_str(char *a, char b[]);
char  *copy_to_gold(char *a, char *b);
char *jump_ept(char *a);

void print_box_coord(box a, FILE *fp, int box);
void print_prob(float *ap, FILE *fp, int classes);
void print_objectness(float a, FILE *fp);
void print_img(double time, int img_num, int box_num, int classes, detection *dets, FILE *fp);
float compare_data_float(float a, float b, float thresh);


#endif

