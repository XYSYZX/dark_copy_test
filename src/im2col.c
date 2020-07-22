#include "im2col.h"
#include <stdio.h>

/*im2col: optimize kernel calculation, 将卷积运算转为矩阵相乘
**        im      net.input  
**        height  输入图像的真正的高度，补0之）
**        width   输入图像的宽度，补0之前
**        channels 输入im的通道数，比如彩色图为3通道，之后每一卷积层的输入的通道数等于上一卷积层卷积核的个数
**        row     要提取的元素所在的行（二维图像补0之后的行数）
**        col     要提取的元素所在的列（二维图像补0之后的列数）
**        channel 要提取的元素所在的通道
**        pad     图像左右上下各补0的长度（四边补0的长度一样）
**  返回： float类型数据，为im中channel通道，row-pad行，col-pad列处的元素值
**  注意：在im中并没有存储补0的元素值，因此height，width都是没有补0时输入图像真正的
**       高、宽；而row与col则是补0之后，元素所在的行列，因此，要准确获取在im中的元素值，
**       首先需要减去pad以获取在im中真实的行列数
*/
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;  //减去补0长度，获取元素真实的行列数
    col -= pad;  

    if (row < 0 || col < 0 || // 如果行列数小于0,则返回0（刚好是补0的效果）
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
/*
 输入： data_im    输入图像
       channels   输入图像的通道数（对于第一层，一般是颜色图，3通道，中间层通道数为上一层卷积核个数）
       height     输入图像的高度（行）
       width      输入图像的宽度（列）
       ksize      卷积核尺寸
       stride     卷积核跨度
       pad        四周补0长度
       data_col   相当于输出，为进行格式重排后的输入图像数据

注:
   data_col还是按行排列，
       行数为channels*ksize*ksize,
       列数为height_col*width_col，即一张特征图总的元素个数，
*/

void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    // 將每一個 kernel 大小的影象轉換成 一列
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;  // 一個 kernel 上的座標 h_offset，w_offset，c_im
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize; 
	// 内循环等于该层输出图像列数width_col，说明最终得到的data_col总有channels_col行，height_col*width_col列
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;   //卷积核上第c个点和输入图像相乘的位置
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

