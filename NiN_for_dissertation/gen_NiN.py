from math import floor
from math import sqrt
import os
#import caffe
from caffe import Net
from caffe import NetSpec
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from IPython import embed
from google.protobuf import text_format
from random import choice
from random import seed
from shutil import copyfile
import conf_firenet as conf
from util_FireNet import mkdir_p
from util_FireNet import save_prototxt
from util_FireNet import FireNet_pooling_layer 
from util_FireNet import FireNet_data_layer
from util_FireNet import get_train_data_layer
from util_FireNet import get_randomized_pooling_scheme
from util_FireNet import conv_relu_xavier
phase='trainval'
#phase='deploy'

'''
    This NiN version is simply for generating geometric examples.

    There are better choices than what we use here for details such as weight initialization.

'''

def NiN_pool(n, layer_str, curr_bottom):
    next_bottom = 'pool' + layer_str
    n.tops[next_bottom] = L.Pooling(n.tops[curr_bottom], kernel_size=3, stride=2, pool=P.Pooling.MAX)
    return next_bottom

#@param opts = list of options [might make this a dict eventually] 
#  dissertation sections ...
#  [TODO]         4.3.1: change number of input channels (3 --> 30)
#  "pool1"    --> 4.3.2: add pool1
#  [TODO]         4.3.3: double the width and height of input data 
#  "moreFilt" --> 4.3.4: 10x filters in [the least comp. intensive layer]
#  "out10k"   --> 4.3.5: 10x more categories of output
def NiN(opts):

    n = NetSpec()
    FireNet_data_layer(n, batch_size) #add data layer to the net
    curr_bottom = 'data'

    #TODO: possibly rename layers to conv1.1, 1.2, 1.3; 2.1, 2.2, etc.

    curr_bottom = conv_relu_xavier(n, 11, 96, str(1), 4, 0, curr_bottom) #_, ksize, nfilt, layerIdx, stride, pad, _
    curr_bottom = conv_relu_xavier(n, 1,  96, str(2), 1, 0, curr_bottom)
    curr_bottom = conv_relu_xavier(n, 1,  96, str(3), 1, 0, curr_bottom)
    curr_bottom = NiN_pool(n, str(3), curr_bottom)

    curr_bottom = conv_relu_xavier(n, 5, 256, str(4), 1, 2, curr_bottom)
    curr_bottom = conv_relu_xavier(n, 1, 256, str(5), 1, 0, curr_bottom)
    curr_bottom = conv_relu_xavier(n, 1, 256, str(6), 1, 0, curr_bottom)
    curr_bottom = NiN_pool(n, str(6), curr_bottom)

    curr_bottom = conv_relu_xavier(n, 3, 384, str(7), 1, 1, curr_bottom)
    curr_bottom = conv_relu_xavier(n, 1, 384, str(8), 1, 0, curr_bottom)
    curr_bottom = conv_relu_xavier(n, 1, 384, str(9), 1, 0, curr_bottom)
    curr_bottom = NiN_pool(n, str(9), curr_bottom)

    curr_bottom = conv_relu_xavier(n, 3, 1024, str(10), 1, 1, curr_bottom)
    curr_bottom = conv_relu_xavier(n, 1, 1024, str(11), 1, 0, curr_bottom)

    n.tops['drop'] = L.Dropout(n.tops[curr_bottom], dropout_ratio=0.5, in_place=True)

    n.tops['conv_12'] = L.Convolution(n.tops[curr_bottom], kernel_size=1, num_output=1000, weight_filler=dict(type='gaussian', std=0.01, mean=0.0))
    n.tops['relu_conv_12'] = L.ReLU(n.tops['conv_12'], in_place=True)
    n.tops['pool_12'] = L.Pooling(n.tops['conv_12'], global_pooling=1, pool=P.Pooling.AVE)

    if phase == 'trainval':
        n.loss = L.SoftmaxWithLoss(n.tops['pool_12'], n.label, include=dict(phase=caffe_pb2.TRAIN))
        n.accuracy = L.Accuracy(n.tops['pool_12'], n.label, include=dict(phase=caffe_pb2.TEST))
        n.accuracy_top5 = L.Accuracy(n.tops['pool_12'], n.label, include=dict(phase=caffe_pb2.TEST), top_k=5) 

    out_dir = 'nets/NiN_' + '_'.join(opts)
    return [n.to_proto(), out_dir]

def save_experiment_config(net_proto, batch_size, out_dir):

    #hack to deal with NetSpec's inability to have two layers both named 'data'
    data_train_proto = get_train_data_layer(batch_size)
    data_train_proto.MergeFrom(net_proto)
    net_proto = data_train_proto

    mkdir_p(out_dir)
    outF = out_dir + '/trainval.prototxt' 
    save_prototxt(net_proto, outF)

    copyfile('solver.prototxt', out_dir + '/solver.prototxt')

    n_gpu = 32
    out_gpu_file = out_dir + '/n_gpu.txt'
    f = open(out_gpu_file, 'w')
    f.write( str(n_gpu) )
    f.close()

if __name__ == "__main__":
    batch_size=1024

    # each of these is one version of NiN.
    NiN_configs = [ [], ['pool1'] ]

    for c in NiN_configs:
        [net_proto, out_dir] = NiN(c)
        save_experiment_config(net_proto, batch_size, out_dir)



