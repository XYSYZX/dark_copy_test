#!/usr/bin/python3.6
import numpy as np
import sys, time

def dec2fp(x, e_bit=8, f_bit=23):
    x = x.astype(np.float64)
    print(x)
    fp_max = 2.**(2.**(e_bit-1)) - 2.**(2.**(e_bit-1)-1-f_bit)   #(2 - 2^-23) * 2^127
    fp_min = 2.**(2. - 2.**(e_bit - 1))                          # 2^(2 - 2^127)
    einf = np.power(2,e_bit) - 1                                 #2^8 - 1
    finf = np.power(2,f_bit) - 1                                 #2^23 - 1
    e_bias = np.power(2,e_bit-1) - 1                             #2^127
    
    x_s = np.where(x > 0, 0, 1).astype(np.int64)	
    x_e_tmp = x_s.copy()
    x_e_tmp[x != 0] = np.floor(np.log2(np.abs(x[x != 0]))).astype(np.int64)
    x_e_tmp[x == 0] = 0
    print(x_e_tmp)
    x_f_tmp = np.rint((np.divide(np.abs(x),2.**x_e_tmp) - 1)*np.power(2,f_bit)).astype(np.int64)
    print(x_f_tmp)
    x_f_rnd = x_f_tmp.copy()
    x_f_rnd[x_f_tmp >= (1 << f_bit)] = x_f_tmp[x_f_tmp >= (1 << f_bit)] - (1 << f_bit)
    x_e_rnd = x_e_tmp.copy()
    x_e_rnd[x_f_tmp >= (1 << f_bit)] = x_e_tmp[x_f_tmp >= (1 << f_bit)] + 1	
    
    x_e_rnd = x_e_rnd + e_bias
    
    x_e = x_e_rnd.copy()
    x_f = x_f_rnd.copy()
    
    x_e[np.abs(x) >= fp_max] = einf - 1
    x_e[np.abs(x)  < fp_min] = 0
    
    x_f[np.abs(x) >= fp_max] = finf
    x_f[np.abs(x)  < fp_min] = 0
    
    out_x = (x_s << (e_bit + f_bit)) + (x_e << f_bit) + x_f
    out_x[x == 0] = 0
    
    return out_x

def fp2dec(x, e_bit=8, f_bit=23):
    
    mask_s = 1 << (e_bit + f_bit)
    mask_e = (np.power(2, e_bit) - 1) << f_bit
    mask_f = np.power(2, f_bit) - 1
    
    
    x_s = np.bitwise_and(mask_s, x)
    x_e = np.bitwise_and(mask_e, x)
    x_f = np.bitwise_and(mask_f, x)
    
    e_bias = x_e.copy()
    e_bias = np.where(e_bias > 0, np.power(2,e_bit-1) - 1, 0)
    
    x_s = np.right_shift(x_s, e_bit + f_bit)
    x_e = np.right_shift(x_e, f_bit) - e_bias
    
    x_f = np.divide(x_f, np.power(2, f_bit))
    
    out_x = (-1)**(x_s)*(1+x_f)*np.power(2.,x_e) 
    mask_0 = 2**(e_bit+f_bit) - 1
    judge_0 = np.bitwise_and(x,mask_0)
    judge_0 = np.where(judge_0 > 0, 1, 0)
    out_x = np.multiply(out_x, judge_0)
    
    return out_x

def main():
    np.random.seed(0)
    bit = 32

    base = 2**np.arange(bit)

    ''' prepare random flip bit sequence 
    '''
    cycle = 100
    percent = 0.1
    i = np.tile(np.arange(cycle), (bit,1))
    for x in range(i.shape[0]):
        np.random.shuffle(i[x])

    i = np.where(i < percent*cycle,1,0).T
    i = np.dot(i,base)

    ''' take a value for testing bit flip 
    '''
    a = np.array([3.2])
    fp_a = dec2fp(a)
    print(fp_a)
    fp_a_flip = np.bitwise_xor(fp_a, i)
#    print(fp2dec(fp_a_flip))

if __name__ == '__main__':
    main()
