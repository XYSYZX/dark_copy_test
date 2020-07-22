#ifndef NOISE_INJECT_H
#define NOISE_INJECT_H

//#define NOISE_FREQ 100000
#define NOISE_FREQ 100

#ifdef GPU
void single_bit_flip_float_gpu(float *w, unsigned int n, int noise_freq);
void single_bit_flip_float_limit_gpu(float *w, unsigned int n, int *limit, int limits, int noise_freq);
#endif
#endif
