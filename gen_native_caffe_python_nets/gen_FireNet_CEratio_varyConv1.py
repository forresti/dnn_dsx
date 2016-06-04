from math import floor
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
phase='trainval'
#phase='deploy'

'''
a FireNet module (similar to inception module) is as follows:

    1x1
   /   \
  3x3  1x1
   \   /
   concat

'''

#@param n = NetSpec object to append (passed by ref, so no need to return it)
#@param bottom = e.g. 'pool1'
#@param layer_idx = e.g. 3 if this is the 3rd FireNet module.
#@return bottom (for next layer) = concat layer string
#TODO: @param number of 1x1_1, 1x1_2, 3x3_2
def FireNet_module(n, bottom, firenet_dict, layer_idx):

    prefix="fire%d/" %layer_idx #e.g. Fire3/squeeze1x1

    #TODO: implement a 'conv_relu' function to reduce lines of code...
    #note: we're doing n.tops[name] instead of n.name, because we like having slashes in our layer names.

    n.tops[prefix+'squeeze1x1'] = L.Convolution(n.tops[bottom], kernel_size=1, num_output=firenet_dict['squeeze1x1_num_output'], weight_filler=dict(type='xavier')) 
    n.tops[prefix+'relu_squeeze1x1'] = L.ReLU(n.tops[prefix+'squeeze1x1'], in_place=True)

    n.tops[prefix+'expand1x1'] = L.Convolution(n.tops[prefix+'squeeze1x1'], kernel_size=1, num_output=firenet_dict['expand1x1_num_output'], weight_filler=dict(type='xavier'))
    n.tops[prefix+'relu_expand1x1'] = L.ReLU(n.tops[prefix+'expand1x1'], in_place=True)

    n.tops[prefix+'expand3x3'] = L.Convolution(n.tops[prefix+'squeeze1x1'], kernel_size=3, pad=1, num_output=firenet_dict['expand3x3_num_output'], weight_filler=dict(type='xavier'))
    n.tops[prefix+'relu_expand3x3'] = L.ReLU(n.tops[prefix+'expand3x3'], in_place=True)

    n.tops[prefix+'concat'] = L.Concat(n.tops[prefix+'expand1x1'], n.tops[prefix+'expand3x3']) #is this right?

    next_bottom = prefix+'concat'
    return next_bottom

    #TODO: perhaps return the concat layer's name, so the next layer can use it as a 'bottom'

def choose_num_output(firenet_layer_idx, s):
    idx=firenet_layer_idx
    firenet_dict = dict()
    #firenet_dict['squeeze1x1_num_output'] = int(s['base_1x1_1'] + floor(idx/s['incr_freq']) * s['incr_1x1_1'])
    firenet_dict['expand1x1_num_output'] = int(s['base_1x1_2'] + floor(idx/s['incr_freq']) * s['incr_1x1_2'])
    firenet_dict['expand3x3_num_output'] = int(s['base_3x3_2'] + floor(idx/s['incr_freq']) * s['incr_3x3_2'])
    firenet_dict['squeeze1x1_num_output'] = int( (firenet_dict['expand1x1_num_output'] + firenet_dict['expand3x3_num_output']) * s['CEratio'])
    return firenet_dict

def FireNet(batch_size, pool_after, s, c1):
    print s

    n = NetSpec()
    FireNet_data_layer(n, batch_size) #add data layer to the net

    layer_idx=1 #e.g. conv1, fire2, etc. 
    n.conv1 = L.Convolution(n.data, kernel_size=c1['dim'], num_output=c1['nfilt'], stride=2, weight_filler=dict(type='xavier'))
    curr_bottom = 'conv1'
    n.tops['relu_conv1'] = L.ReLU(n.tops[curr_bottom], in_place=True)

    #if curr_bottom in pool_after.keys():
    #    curr_bottom = FireNet_pooling_layer(n, curr_bottom, pool_after[curr_bottom], layer_idx)

    if layer_idx in pool_after:
        n.tops['pool1'] = L.Pooling(n.tops[curr_bottom], kernel_size=3, stride=2, pool=P.Pooling.MAX)
        curr_bottom = 'pool1'    

    for layer_idx in xrange(2, s['n_layers']+2):
        firenet_dict = choose_num_output(layer_idx-2, s)
        print firenet_dict
        curr_bottom = FireNet_module(n, curr_bottom, firenet_dict, layer_idx) 

        if layer_idx in pool_after:
            next_bottom = 'pool%d' %layer_idx
            n.tops[next_bottom] = L.Pooling(n.tops[curr_bottom], kernel_size=3, stride=2, pool=P.Pooling.MAX)
            curr_bottom = next_bottom

    n.tops['drop'+str(layer_idx)] = L.Dropout(n.tops[curr_bottom], dropout_ratio=0.5, in_place=True)

    #optional pre_conv_final (w/ appropriate CEratio)
    #n.pre_conv_final = L.Convolution(n.tops[curr_bottom], kernel_size=1, num_output=int(1000*s['CEratio']), stride=1, weight_filler=dict(type='xavier'))
    #n.tops['relu_pre_conv_final'] = L.ReLU(n.tops['pre_conv_final'], in_place=True)
    #curr_bottom='pre_conv_final'

    n.tops['conv_final'] = L.Convolution(n.tops[curr_bottom], kernel_size=1, num_output=1000, weight_filler=dict(type='gaussian', std=0.01, mean=0.0)) 
    n.tops['relu_conv_final'] = L.ReLU(n.tops['conv_final'], in_place=True) 
    n.tops['pool_final'] = L.Pooling(n.tops['conv_final'], global_pooling=1, pool=P.Pooling.AVE)
 
    if phase == 'trainval':
        n.loss = L.SoftmaxWithLoss(n.tops['pool_final'], n.label, include=dict(phase=caffe_pb2.TRAIN))
        n.accuracy = L.Accuracy(n.tops['pool_final'], n.label, include=dict(phase=caffe_pb2.TEST))
        n.accuracy_top5 = L.Accuracy(n.tops['pool_final'], n.label, include=dict(phase=caffe_pb2.TEST), top_k=5) 
    return n.to_proto()

def get_pooling_schemes_new():
    p = []
    p.append({1,4,8})
    p.append({1,2,4,8})
    p.append({1,3,6})
    p.append({1,3,5})
    p.append({1,3,5,7})
    return p

def get_base_incr_schemes():
    base_incr = []
    #CEratio = 0.5 # (1x1_1) / (1x1_2 + 3x3_2)
    #n_layers is number of fire layers.

    #for CEratio in [0.125, 0.25, 0.5, 0.75, 1.0]:
    #for CEratio in [0.125, .150, 0.175, .200, .225, .250]:
    #for CEratio in [0.125, 0.175]:
    #for CEratio in [0.125]:
    #    base_incr.append({'base_1x1_2':64,  'base_3x3_2':64, 'incr_1x1_2':64, 'incr_3x3_2':64, 'CEratio':CEratio, 'incr_freq':2, 'n_layers':8})


    for CEratio in [0.125, 0.25, 0.5, 0.75, 1.0]:
        #base_incr.append({'base_1x1_2':48,  'base_3x3_2':48, 'incr_1x1_2':48, 'incr_3x3_2':48, 'CEratio':CEratio, 'incr_freq':2, 'n_layers':8})
        #base_incr.append({'base_1x1_2':32,  'base_3x3_2':32, 'incr_1x1_2':32, 'incr_3x3_2':32, 'CEratio':CEratio, 'incr_freq':2, 'n_layers':8})
        base_incr.append({'base_1x1_2':64,  'base_3x3_2':64, 'incr_1x1_2':64, 'incr_3x3_2':64, 'CEratio':CEratio, 'incr_freq':2, 'n_layers':8})

    '''
    CEratio = .75
    base_incr.append({'base_1x1_2':64,  'base_3x3_2':64, 'incr_1x1_2':128, 'incr_3x3_2':64, 'CEratio':CEratio, 'incr_freq':2})
    base_incr.append({'base_1x1_2':64,  'base_3x3_2':64, 'incr_1x1_2':128, 'incr_3x3_2':128, 'CEratio':CEratio, 'incr_freq':2})
    base_incr.append({'base_1x1_2':64,  'base_3x3_2':64, 'incr_1x1_2':192, 'incr_3x3_2':64, 'CEratio':CEratio, 'incr_freq':2})
    base_incr.append({'base_1x1_2':16,  'base_3x3_2':64, 'incr_1x1_2':192, 'incr_3x3_2':64, 'CEratio':CEratio, 'incr_freq':2})
    '''
    return base_incr

def get_conv1_schemes():
    conv1 = []

    #for stride in [2,3]:
    #for dim in [2,3,5,7,9]:
    dim=3
    #for nfilt in [16, 32, 64, 96]:
    for nfilt in [64]:
        conv1.append({'dim':dim, 'nfilt':nfilt})
    return conv1

if __name__ == "__main__":
    batch_size=512

    #p = {1,4,8}
    #p = {1,2,4,8}
    #p = {1,3,5,7}
    #pooling_schemes = get_pooling_schemes_new()
    #pooling_schemes = [{1,4,8}, {1,3,6}]
    #pooling_schemes = [{1,3,6}, {1,3,5}]
    #pooling_schemes = [{1,3,5}, {1,3,5,9}, {1,3,5,8}]
    pooling_schemes = [{1,3,5}]
    conv1_schemes = get_conv1_schemes()

    for c1 in conv1_schemes:
        for p in pooling_schemes:
            pstr = '_'.join([str(x) for x in sorted(p)]) #[1,2,4] -> '1_2_4'

            base_incr_schemes = get_base_incr_schemes()
            for s in base_incr_schemes:
                net_proto = FireNet(batch_size, p, s, c1)

                #hack to deal with NetSpec's inability to have two layers both named 'data'
                data_train_proto = get_train_data_layer(batch_size)
                data_train_proto.MergeFrom(net_proto)
                net_proto = data_train_proto

                out_dir = 'nets/%s_FireNet_%d_fireLayers_batch_%d_base_r_%d_%d_incr_r_%d_%d_CEratio_%0.3f_freq_%d_pool_%s_conv1_%d_dim_%d_filt' %(conf.dataset, s['n_layers'], batch_size, s['base_1x1_2'], s['base_3x3_2'],
                                                                                            s['incr_1x1_2'], s['incr_3x3_2'], s['CEratio'], s['incr_freq'], pstr, c1['dim'], c1['nfilt'])

                mkdir_p(out_dir)
                #eoutF = 'FireNet_pool_%s.prototxt' %p #e.g. pool_early
                outF = out_dir + '/trainval.prototxt' 
                save_prototxt(net_proto, outF)

                copyfile('gentle_solver.prototxt', out_dir + '/solver.prototxt')

                n_gpu = 32
                out_gpu_file = out_dir + '/n_gpu.txt'
                f = open(out_gpu_file, 'w')
                f.write( str(n_gpu) )
                f.close()

