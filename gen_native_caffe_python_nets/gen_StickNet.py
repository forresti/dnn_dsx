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
this is all there is to StickNet:
    1x1 -> pool -> 1x1 -> 1x1 -> pool -> 1x1 -> (etc)
'''

#thx: http://stackoverflow.com/questions/4265546/python-round-to-nearest-05
def round_to(n, precision):
    return int(round(n / precision) * precision)

#given other dims and a target number of FLOPS per image, select the number of filters (i.e. num_output)
def choose_num_output(filterH, filterW, ch, activH, activW, mflop_per_img_target, n_layers):
    #TODO: remove 'ch' from input args, since it's unused

    #goal is for each layer to have (mflop_per_img_target / n_layers) FLOPS.
    mflop_per_layer_target = float(mflop_per_img_target)/n_layers

    flop_per_filt = filterH * filterW * activH * activW * 2 #not counting channels or nfilt
    mflop_per_filt = flop_per_filt / 1e6

    #assume: for several layers downstream of this one, we'll have nfilt_{i-1} = nfilt_i. 
    #so, we aim for [flop_per_filt * nfilt^2 == mflop_per_img_target]
    #           --> [nfilt = sqrt(mflop_per_img_target / flop_per_filt)

    nfilt = sqrt(mflop_per_layer_target / mflop_per_filt)

    selected_mflop = nfilt * nfilt * flop_per_filt / 1e6

    print "  mflop_per_layer=%f, mflop_per_img=%f, mflop_per_filt=%f, selected_nfilt=%f, selected_mflop=%f" %(mflop_per_layer_target, mflop_per_img_target, mflop_per_filt, nfilt, selected_mflop) 

    #TODO: round to nearest 8
    return nfilt

#rough estimate of activation size ... ignoring padding and other nuances.
def est_activ_size(inImgH, inImgW, totalStride):
    activH = inImgH/totalStride
    activW = inImgW/totalStride
    return [activH, activW]

'''
TODO:
1.   [done] calculate n_filt based on FLOP goal
1.1. [done] keep track of total_stride (for selecting n_filt)  
1.2. round n_filt to nearest 8
2.   name layers conv1_1, conv1_2, pool1, conv2_1, ... pool2, conv3_1, etc.
3.   [done] gen pool layers 
'''
def StickNet(batch_size, s):
    inImgH = 224 #TODO: put inImg{H,W} into 's' if necessary.
    inImgW = 224
    round_to_nearest = 4

    n = NetSpec()
    FireNet_data_layer(n, batch_size) #add data layer to the net
    curr_bottom='data'

    #layer-to-layer counters
    _totalStride = 1 #note that, using 1x1 conv, our (stride>1) is only in pooling layers.
    _numPoolings = 1 #for indexing 'conv2_1', etc.
    _ch=3
    [activH, activW] = est_activ_size(inImgH, inImgW, _totalStride)
    n_filt = choose_num_output(1, 1, _ch, activH, activW, s['mflop_per_img_target'], s['n_layers']) #only using this for conv1 to avoid oscillations.
    n_filt = round_to(n_filt, round_to_nearest) #make divisible by 8

    #FIXME: somehow account for num_output produced by conv1 when selecting number of filters for conv2. (else, conv2 goes way over budget on flops.)
    # perhaps we need to find the number N such that N^2*activations = mflop_per_img_target?

    idx_minor = 1
    idx_major = 1

    #this goes to (n_layers-1) ... then we do conv_final separately because it has a different weight init.
    for layer_idx in xrange(1, s['n_layers']):

        layer_str = '%d.%d' %(idx_major, idx_minor)

        #select number of filters in this layer:
        #[activH, activW] = est_activ_size(inImgH, inImgW, _totalStride)
        #n_filt = choose_num_output(1, 1, _ch, activH, activW, s['mflop_per_img_target'], s['n_layers']) 
        #TODO: to avoid oscillations, perhaps just use choose_num_output for conv1, 
        #      and then just double n_filt whenever we do stride=2.

        #generate layer
        ksize=1
        stride=1
        pad=0
        curr_bottom = conv_relu_xavier(n, ksize, n_filt, layer_str, stride, pad, curr_bottom)
        _ch = n_filt #for next layer

        if layer_idx in s['pool_after'].keys():
            pinfo = s['pool_after'][layer_idx]

            #next_bottom = 'pool%d' %layer_idx
            next_bottom = 'pool_' + layer_str
            n.tops[next_bottom] = L.Pooling(n.tops[curr_bottom], kernel_size=pinfo['kernel_size'], stride=pinfo['stride'], pool=P.Pooling.MAX)
            curr_bottom = next_bottom

            _totalStride = _totalStride * pinfo['stride']
            _numPoolings = _numPoolings + 1

            n_filt = n_filt * pinfo['stride'] #to keep (most) layers at roughly the same complexity-per-layer

            idx_major = idx_major + 1
            idx_minor = 1

        else:
            idx_minor = idx_minor + 1

    n.tops['drop'+str(layer_idx)] = L.Dropout(n.tops[curr_bottom], dropout_ratio=0.5, in_place=True)

    n.tops['conv_final'] = L.Convolution(n.tops[curr_bottom], kernel_size=1, num_output=1000, weight_filler=dict(type='gaussian', std=0.01, mean=0.0)) 
    n.tops['relu_conv_final'] = L.ReLU(n.tops['conv_final'], in_place=True) 
    n.tops['pool_final'] = L.Pooling(n.tops['conv_final'], global_pooling=1, pool=P.Pooling.AVE)
 
    if phase == 'trainval':
        n.loss = L.SoftmaxWithLoss(n.tops['pool_final'], n.label, include=dict(phase=caffe_pb2.TRAIN))
        n.accuracy = L.Accuracy(n.tops['pool_final'], n.label, include=dict(phase=caffe_pb2.TEST))
        n.accuracy_top5 = L.Accuracy(n.tops['pool_final'], n.label, include=dict(phase=caffe_pb2.TEST), top_k=5) 
    return n.to_proto()

#default pool config
def dp():
    p={'stride':2, 'kernel_size':3}
    return p

def get_base_incr_schemes():
    base_incr = []
    #0.5 TF = 500,000,000,000 (per batch of 1024)
    #per img, we want: 500,000,000 FLOPS = 500 MFLOPS.

    #mflop_per_img_target = 500 #out of memory ... going smaller.
    #base_incr.append({'mflop_per_img_target':mflop_per_img_target, 'n_layers':20, 'pool_after':{1:dp(), 5:dp(), 9:dp(), 13:dp()}})

    for mflop_per_img_target in [100, 250, 500]:
        for n_layers in [8, 10, 15, 20]:
            base_incr.append({'mflop_per_img_target':mflop_per_img_target, 'n_layers':n_layers, 'pool_after':{1:dp(), 5:dp(), 9:dp(), 13:dp()}})

    #TODO: optional 'conv1_override'

    return base_incr

#s = one scheme from get_base_incr_schemes()
#@return name of directory that describes this experiment.
def scheme_to_fname(s):
    #p = [pa['layer_idx'] for pa in s['pool_after']]
    p = s['pool_after'].keys()
    pstr = '_'.join([str(x) for x in sorted(p)]) #[1,2,4] -> '1_2_4'

    #st = 'nets/StickNet_warmup_%d_layers_%d_flops_pool_%s' %(s['n_layers'], s['mflop_per_img_target'], pstr)
    st = 'nets/StickNet_warmup_%d_mflops_%d_layers_pool_%s' %(s['mflop_per_img_target'], s['n_layers'], pstr)
    return st

if __name__ == "__main__":
    batch_size=512

    base_incr_schemes = get_base_incr_schemes()
    for s in base_incr_schemes:

        net_proto = StickNet(batch_size, s)

        #hack to deal with NetSpec's inability to have two layers both named 'data'
        data_train_proto = get_train_data_layer(batch_size)
        data_train_proto.MergeFrom(net_proto)
        net_proto = data_train_proto

        out_dir = scheme_to_fname(s)
 
        mkdir_p(out_dir)
        outF = out_dir + '/trainval.prototxt' 
        save_prototxt(net_proto, outF)

        copyfile('warmup_solver.prototxt', out_dir + '/solver.prototxt')

        n_gpu = 16
        out_gpu_file = out_dir + '/n_gpu.txt'
        f = open(out_gpu_file, 'w')
        f.write( str(n_gpu) )
        f.close()

