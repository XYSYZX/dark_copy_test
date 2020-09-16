#include "bit_flip.h"
#include <stdlib.h>
#include <stdio.h>

unsigned int *dec2bin(float fval, unsigned int *uBin)
{
    unsigned char ucArray[4];
    unsigned char *puc;
    int float_len = sizeof(float);
    puc = (unsigned char *)&fval;
    for(int i=0; i<float_len; i++){
        ucArray[i] = puc[i];
    }
    for(int i=0; i<float_len; i++){
        for(int j=0; j<8; j++){
            if((ucArray[float_len-1-i])&0x1<<(7-j))
                uBin[8*i+j] = 1;
            else
                uBin[8*i+j] = 0;
        }
    }
    return uBin;
}

void print_decbin(int *uBin, int length)
{
    for(int i=0; i<length; i++)
        printf("%d", uBin[i]);
    printf("\n");
}
/*
float bin2dec(int *uBin)
{
    float res;
    unsigned char *puc = (unsigned char *)&res;
}
*/
float *single_bit_flip_float(float *w, int idx)
{
    idx = idx%32;
    unsigned int b = 0x1;
    unsigned int *p;
    p = (unsigned int *)w;
    (*p) = (*p)^(b<<idx);
    return w;
}

/*
int main(int argc, char *argv[])
{
    if(argc<=1){
        printf("dont know which bit to flip\n");
        return 0;
    }
//    int idx = argv[1] - '0';
//    int idx = atoi(argv[2]);
    float w = atof(argv[1]);
    float tmp = w;
    unsigned int uBinb[8*sizeof(float)];
    printf("before: %f\n", w);
    dec2bin(w, uBinb);
    print_decbin(uBinb, 8*sizeof(float));
    for(int idx=0; idx < 33; idx++){
        if(idx == 31)
            continue;
        single_bit_flip_float(&w, idx);
        printf("after: %f\n", w);
        dec2bin(w, uBinb);
        print_decbin(uBinb, 8*sizeof(float));
        w = tmp;
    }
    return 0;
}
*/
