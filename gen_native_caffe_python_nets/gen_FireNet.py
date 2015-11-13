#import caffe
from caffe import Net
from caffe import NetSpec
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from IPython import embed
from google.protobuf import text_format

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

def save_prototxt(protobuf, out_fname):
    f = open(out_fname, 'w')
    f.write(text_format.MessageToString(protobuf))
    f.flush()
    f.close()

#@param n = NetSpec object to append (passed by ref, so no need to return it)
#@param bottom = e.g. 'pool1'
#@param layer_idx = e.g. 3 if this is the 3rd FireNet module.
#@return bottom (for next layer) = concat layer string
#TODO: @param number of 1x1_1, 1x1_2, 3x3_2
def FireNet_module(n, bottom, firenet_dict, layer_idx):

    prefix="fire%d/" %layer_idx #e.g. Fire3/conv1x1_1

    #TODO: implement a 'conv_relu' function to reduce lines of code...
    #note: we're doing n.tops[name] instead of n.name, because we like having slashes in our layer names.

    n.tops[prefix+'conv1x1_1'] = L.Convolution(n.tops[bottom], kernel_size=1, num_output=firenet_dict['conv1x1_1_num_output'], weight_filler=dict(type='xavier')) 
    n.tops[prefix+'relu_conv1x1_1'] = L.ReLU(n.tops[prefix+'conv1x1_1'], in_place=True)

    n.tops[prefix+'conv1x1_2'] = L.Convolution(n.tops[prefix+'conv1x1_1'], kernel_size=1, num_output=firenet_dict['conv1x1_2_num_output'], weight_filler=dict(type='xavier'))
    n.tops[prefix+'relu_conv1x1_2'] = L.ReLU(n.tops[prefix+'conv1x1_2'], in_place=True)

    n.tops[prefix+'conv3x3_2'] = L.Convolution(n.tops[prefix+'conv1x1_1'], kernel_size=3, pad=1, num_output=firenet_dict['conv3x3_2_num_output'], weight_filler=dict(type='xavier'))
    n.tops[prefix+'relu_conv3x3_2'] = L.ReLU(n.tops[prefix+'conv3x3_2'], in_place=True)

    n.tops[prefix+'concat'] = L.Concat(n.tops[prefix+'conv1x1_2'], n.tops[prefix+'conv3x3_2']) #is this right?

    next_bottom = prefix+'concat'
    return next_bottom

    #TODO: perhaps return the concat layer's name, so the next layer can use it as a 'bottom'

def FireNet_pooling_layer(n, bottom, pool_spec, layer_idx):
    p = pool_spec
    next_bottom='pool'+str(layer_idx)
    n.tops[next_bottom] = L.Pooling(n.tops[bottom], kernel_size=p['kernel_size'], 
                                                stride=p['stride'], pool=p['pool']) 
    return next_bottom

def choose_num_output(firenet_layer_idx):
    firenet_dict = dict()
    firenet_dict['conv1x1_1_num_output'] = 128 + firenet_layer_idx*128
    firenet_dict['conv3x3_2_num_output'] = 64 + firenet_layer_idx*64
    firenet_dict['conv1x1_2_num_output'] = 128 + firenet_layer_idx*128 
    return firenet_dict

#@param NetSpec n
def FireNet_data_layer(n, batch_size):
    #TODO: set phase...
    test_lmdb='/rscratch/forresti/ILSVRC12/ilsvrc2012_val_256x256_lmdb/'
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=test_lmdb,
                             transform_param=dict(crop_size=227, mean_value=[104, 117, 123]), ntop=2)

def FireNet(batch_size):

    #TODO: take pool_after as an input argument.
    pool_after = {'conv1':{'kernel_size':3, 'stride':2, 'pool':P.Pooling.MAX},
                  'fire3/concat': {'kernel_size':3, 'stride':2, 'pool':P.Pooling.MAX},
                  'fire4/concat': {'kernel_size':3, 'stride':2, 'pool':P.Pooling.MAX}} 

    n = NetSpec()
    FireNet_data_layer(n, batch_size) #add data layer to the net

    layer_idx=1 #e.g. conv1, fire2, etc. 
    n.conv1 = L.Convolution(n.data, kernel_size=7, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    curr_bottom = 'conv1'
    n.tops['relu_conv1'] = L.ReLU(n.tops[curr_bottom], in_place=True)

    if curr_bottom in pool_after.keys():
        curr_bottom = FireNet_pooling_layer(n, curr_bottom, pool_after[curr_bottom], layer_idx)
    
    for layer_idx in xrange(2,6):
        firenet_dict = choose_num_output(layer_idx-2)
        print firenet_dict
        curr_bottom = FireNet_module(n, curr_bottom, firenet_dict, layer_idx) 

        if curr_bottom in pool_after.keys():
            curr_bottom = FireNet_pooling_layer(n, curr_bottom, pool_after[curr_bottom], layer_idx) 


    n.tops['conv_final'] = L.Convolution(n.tops[curr_bottom], kernel_size=1, pad=1, num_output=1000, weight_filler=dict(type='xavier'))   
    n.tops['relu_conv_final'] = L.ReLU(n.tops['conv_final'], in_place=True) 
    n.tops['pool_final'] = L.Pooling(n.tops['conv_final'], global_pooling=1, pool=P.Pooling.AVE)
 
    if phase == 'trainval':
        n.loss = L.SoftmaxWithLoss(n.tops['pool_final'], n.label, include=dict(phase=caffe_pb2.TRAIN))
        n.accuracy = L.Accuracy(n.tops['pool_final'], n.label, include=dict(phase=caffe_pb2.TEST)) 
    return n.to_proto()

#get protobuf containing only a data layer
def gen_singleton_data_layer(batch_size, phase):
    #TODO: set phase in protobuf.
    #TODO: select LMDB based on phase
    n = NetSpec()
    test_lmdb='/rscratch/forresti/ILSVRC12/ilsvrc2012_val_256x256_lmdb/'
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=test_lmdb,
                             transform_param=dict(crop_size=227, mean_value=[104, 117, 123]), ntop=2)
    return n.to_proto()

if __name__ == "__main__":

    batch_size=128
    net_proto = FireNet(batch_size)

    #TODO: make 'train' in FireNet(), and add 'test' here
    data_train_proto = gen_singleton_data_layer(batch_size, 'train')
    data_train_proto.MergeFrom(net_proto)
    net_proto = data_train_proto

    save_prototxt(net_proto, 'FireNet.prototxt')



