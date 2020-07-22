#!/usr/bin/python3

import argparse
import os
import numpy as np


def parser():
    parser = argparse.ArgumentParser(description="Darknet\'s yolov3 weights")
    parser.add_argument('-weights_path', help='yolov3 weights path')
    parser.add_argument('-cfg_path', help='yolov3.cfg')
#    parser.add_argument('-output_path', help='file of bit flipped weights')
    return parser.parse_args()

class weightLoader(object):
    
    def __init__(self, weights_path):
        try:
            self.fhandle = open(weights_path, 'rb')
        except IOError:
            print('fail to open weight file!')
        self.read_bytes = 0

    def parser_buffer(self, shape, dtype='int32', buffer_size=None):        #用来读文件内容
        self.read_bytes += buffer_size
        return np.ndarray(shape=shape, dtype=dtype, buffer=self.fhandle.read(buffer_size))

    def head(self):         #权重文件的前160个字节存储5个int32值，构成文件的头部
        major, minor, revision = self.parser_buffer(shape=(3,), dtype='int32', buffer_size=12)
        if major*10 + minor >=2 and major < 1000 and minor < 100:
            seen = self.parser_buffer(shape=(1,), dtype='int64',buffer_size=8)
        else:
            seen = self.parser_buffer(shape=(1,), dtype='int32',buffer_size=4)
        return major, minor, revision, seen

    def conv(self, filters, weight_size, batch_normalize):
        print("loading: bias: %d, weights: %d"  %(filters, weight_size))
        conv_bias = self.parser_buffer(shape=(filters,), dtype='float32', buffer_size=filters*4)
        if batch_normalize:
            scale, mean, var = self.bn(filters)
            batch_norm = [scale, mean, var]
        else:
            batch_norm = []
        conv_weights = self.parser_buffer(shape=(weight_size,), dtype='float32', buffer_size=weight_size*4)
        return conv_bias, conv_weights, batch_norm
            
    def bn(self,filters):
        '''
        bn有4个参数，分别是bias，scale，mean，var，
          其中bias已经读取完毕，这里读取剩下三个，scale,mean,var 
        '''
        bn_weights = self.parser_buffer(
                              shape=(3,filters),
                              dtype='float32',
                              buffer_size=(filters*3)*4)
        # scale, mean,var
        return bn_weights[0], bn_weights[1], bn_weights[2]

    def close(self):
        self.fhandle.close()

class parse_cfg(object):
    def __init__(self, cfg_path):

        self.block_gen = self._get_block(cfg_path)
#        self.weight_loader = weightLoader(weights_path)
        self.prev_layer = {'filters': 3}     #input images has 3 channel, aka filters
#        self.input_layer = Input(shape=(None, None, 3))
#        self.out_index = []
#        self.prev_layer = self.input_layer
#        self.layer_num = 0
        self.layer_num = -2
        self.all_layers = []
        self.bias_weights = []
        self.batch_normalize = []
#        self.count = [0,0]

    def _get_block(self, cfg_path):
        block = {}
        with open(cfg_path, 'r') as fr:
            for line in fr:
                line = line.strip()
                if '[' in line and ']' in line:
                    if block:
                        self.layer_num += 1
                        yield block
                    block = {}
                    block['type'] = line.strip(' []')
                elif not line or '#' in line:
                    continue
                else:
                    key, val = line.strip().replace(' ', '').split('=')
                    key, val = key.strip(), val.strip()
                    block[key] = val
            self.layer_num += 1
            yield block

    def conv_par(self, block):
#        self.count[0] += 1
        filters = int(block['filters'])
        size = int(block['size'])
#        stride = int(block['stride'])
#        pad = int(block['pad'])
#        activation = block['activation']

#       padding = size/2 if pad == 1
#        batch_normalize = 'batch_normalize' in block
        if 'batch_normalize' in block:
            batch_normalize = block['batch_normalize']
        else:
            batch_normalize = 0
        weights_size = self.prev_layer['filters']*filters*size*size
        print("layer: %d, weights: %d" %(self.layer_num, weights_size))
        self.all_layers.append(filters)
        self.prev_layer['filters'] = filters
        self.bias_weights.append([filters, weights_size])
        self.batch_normalize.append(batch_normalize)
        return weights_size

    def route_par(self, block):
        filters = 0
        if ',' in block['layers']:
            layers = [int(x.strip()) for x in block['layers'].split(',')]
        else:
            layers = [int(block['layers'].strip())]
        for lay in layers:
            if lay < 0:
                index = self.layer_num + lay
            else:
                index = lay
            print(index)
            filters += self.all_layers[index]   #get channel
        self.all_layers.append(filters)
        self.prev_layer['filters'] = filters

    def other_par(self, block):
            if block['type'] == 'net':
                return
            self.all_layers.append(self.prev_layer['filters'])

    def par_all(self,):

        for idx,block in enumerate(self.block_gen):
            if block['type'] == 'convolutional':
                self.conv_par(block)
            elif block['type'] == 'route':
                cfg_parser.route_par(block)
            else:
                cfg_parser.other_par(block)

    """    
    def maxpool(self,block):
        size = int(block['size'])
        stride = int(block['stride'])
        maxpool_layer = MaxPooling2D(pool_size=(size,size),
                        strides=(stride,stride),
                        padding='same')(self.prev_layer)
        self.all_layers.append(maxpool_layer)
        self.prev_layer = maxpool_layer

    def shortcut(self,block):
        index = int(block['from'])
        activation = block['activation']
        assert activation == 'linear', 'Only linear activation supported.'
        shortcut_layer = Add()([self.all_layers[index],self.prev_layer])
        self.all_layers.append(shortcut_layer)
        self.prev_layer = shortcut_layer
    def route(self,block):
        layers_ids = block['layers']
        ids = [int(i) for i in layers_ids.split(',')]
        layers = [self.all_layers[i] for i in ids]
        if len(layers) > 1:
            print('Concatenating route layers:', layers)
            concatenate_layer = Concatenate()(layers)
            self.all_layers.append(concatenate_layer)
            self.prev_layer = concatenate_layer
        else:
            skip_layer = layers[0]
            self.all_layers.append(skip_layer)
            self.prev_layer = skip_layer

    def upsample(self,block):
        stride = int(block['stride'])
        assert stride == 2, 'Only stride=2 supported.'
        upsample_layer = UpSampling2D(stride)(self.prev_layer)
        self.all_layers.append(upsample_layer)
        self.prev_layer = self.all_layers[-1]

    def yolo(self,block):
        self.out_index.append(len(self.all_layers)-1)
        self.all_layers.append(None)
        self.prev_layer = self.all_layers[-1]

    def net(self, block):
        self.weight_decay = block['decay']

    def create_and_save(self,weights_only,output_path):
        if len(self.out_index) == 0:
            self.out_index.append( len(self.all_layers)-1 )

        output_layers = [self.all_layers[i] for i in self.out_index]
        model = Model(inputs=self.input_layer,
                      outputs=output_layers)
        print(model.summary())

        if weights_only:
            model.save_weights(output_path)
            print('Saved Keras weights to {}'.format(output_path))
        else:
            model.save(output_path)
            print('Saved Keras model to {}'.format(output_path))
    def close(self):
        self.weight_loader.close()
    """

def main():
    args = parser()
    print('Parsing Darknet config.')
    cfg_parser = parse_cfg(args.cfg_path)
    for idx,block in enumerate(cfg_parser.block_gen):
        if block['type'] == 'convolutional':
            weights_size = cfg_parser.conv_par(block)
        elif block['type'] == 'route':
            cfg_parser.route_par(block)
        else:
            cfg_parser.other_par(block)
    
    weight_loader = weightLoader(args.weights_path)
    i = 0
    conv_layers = len(cfg_parser.bias_weights)
    major, minor, revision, seen = weight_loader.head()
    while i < conv_layers:
        conv_bias, conv_weights, batch_norm = weight_loader.conv(cfg_parser.bias_weights[i][0], cfg_parser.bias_weights[i][1], cfg_parser.batch_normalize[i])   #filters, weights, batch_norm
        i += 1
    weight_loader.close() 

            
if __name__ == '__main__':
    main()
