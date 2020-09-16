#include "noise_inject.h"
#include "darknet.h"
#include "bit_flip.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

//#define NOISE_FREQ 1

void inject_noise_float(float *w, unsigned int length)
{
    int count = 0;
    int noise_freq = NOISE_FREQ;
    int idx;
    int t;
    srand((unsigned)time(NULL));
    for(int i=0; i<length; i++){
        if(count==0){
            t = (int)(rand()/(double)RAND_MAX + 1)*noise_freq;
        }
        if(count == t){
            count = 0;
            idx = rand()%33;   //get num from 0 to 32
            single_bit_flip_float(w, idx);
            continue;
        }
        count++;
        w++;
    }
}

void inject_noise_float_limit(float *w, unsigned int length, int *limit, int limits)
{
    int count = 0; 
    int noise_freq = NOISE_FREQ;
    int idx;
    int t;
    srand((unsigned)time(NULL));
    for(int i=0; i<length; i++){
        if(count==0)
            t = (int)(rand()/(double)RAND_MAX + 1)*noise_freq;
        if(count == t){
            int flag = 0;
            idx = rand()%33;   //get num from 0 to 32
            for(int j=0; j<limits; j++){
                if(idx==limit[j])
                    flag = 1;
            }
            if(flag == 0)
                single_bit_flip_float(w, idx);
            else
                continue;
            count = 0;
            continue;
        }
        count++;
        w++;
    }
}

void inject_noise_float_onebit(float *w, int idx, int bit_idx)
{
	float *x = &w[idx];
	single_bit_flip_float(x, bit_idx);
}

void inject_noise_float_manybit(float *w, int idx, int bit_len, int *bit_idxs)
{
	float *x = &w[idx];
    for(int i = 0; i < bit_len; i++){
	    single_bit_flip_float(x, bit_idxs[i]);
    }
}

