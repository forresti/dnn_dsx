#import caffe
from caffe import Net
from caffe import NetSpec
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from IPython import embed
from google.protobuf import text_format

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
#TODO: @param if/when to pool
#TODO: @param number of 1x1_1, 1x1_2, 3x3_2
def FireNet_module(n, bottom, layer_idx):

    prefix="fire%d/" %layer_idx #e.g. Fire3/conv1x1_1

    #TODO: implement a 'conv_relu' function to reduce lines of code...

    #note: we're doing n.tops[name] instead of n.name, because we like having slashes in our layer names.

    n.tops[prefix+'conv1x1_1'] = L.Convolution(n.tops[bottom], kernel_size=1, num_output=128, weight_filler=dict(type='xavier')) 
    n.tops[prefix+'relu_conv1x1_1'] = L.ReLU(n.tops[prefix+'conv1x1_1'], in_place=True)

    n.tops[prefix+'conv1x1_2'] = L.Convolution(n.tops[prefix+'conv1x1_1'], kernel_size=1, num_output=128, weight_filler=dict(type='xavier'))
    n.tops[prefix+'relu_conv1x1_2'] = L.ReLU(n.tops[prefix+'conv1x1_2'], in_place=True)

    n.tops[prefix+'conv3x3_2'] = L.Convolution(n.tops[prefix+'conv1x1_1'], kernel_size=3, pad=1, num_output=128, weight_filler=dict(type='xavier'))
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

def FireNet(batch_size):

    #TODO: take pool_after as an input argument.
    pool_after = {'conv1':{'kernel_size':3, 'stride':2, 'pool':P.Pooling.MAX},
                  'fire3/concat': {'kernel_size':3, 'stride':2, 'pool':P.Pooling.MAX}} 

    n = NetSpec()
    n.data, n.label = L.DummyData(shape=[dict(dim=[batch_size, 1, 28, 28]),
                                         dict(dim=[batch_size, 1, 1, 1])],
                                  transform_param=dict(scale=1./255), ntop=2)

    layer_idx=1 #e.g. conv1, fire2, etc. 
    n.conv1 = L.Convolution(n.data, kernel_size=7, num_output=96, stride=2, weight_filler=dict(type='xavier'))
    curr_bottom = 'conv1'

    #TODO: increment #filters in here...
    for layer_idx in xrange(2,4):
        curr_bottom = FireNet_module(n, curr_bottom, layer_idx) #should create Fire2/...

        if curr_bottom in pool_after.keys():
            curr_bottom = FireNet_pooling_layer(n, curr_bottom, pool_after[curr_bottom], layer_idx) 

    n.loss = L.SoftmaxWithLoss(n.tops[curr_bottom], n.label, include=dict(phase=caffe_pb2.TRAIN)) 
    n.accuracy = L.Accuracy(n.tops[curr_bottom], n.label, include=dict(phase=caffe_pb2.TEST)) 
    return n.to_proto()

if __name__ == "__main__":
    #net_proto = lenet(128)
    #save_prototxt(net_proto, 'lenet.prototxt')

    net_proto = FireNet(128)
    save_prototxt(net_proto, 'FireNet.prototxt')



