#ifndef BOUND_ADJUST_H
#define BOUND_ADJUST_H


void constrain_cpu(float *values, int max, int min, int nums);
#ifdef GPU
//void constrain_gpu(float *values, int max, int min, int nums);
#endif
#endif
