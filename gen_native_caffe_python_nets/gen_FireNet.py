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

    prefix="Fire%d/" %layer_idx #e.g. Fire3/conv1x1_1

    #TODO: implement a 'conv_relu' function to reduce lines of code...

    #note: we're doing n.tops[name] instead of n.name, because we like having slashes in our layer names.

    n.tops[prefix+'conv1x1_1'] = L.Convolution(n.tops[bottom], kernel_size=1, num_output=128, weight_filler=dict(type='xavier')) 
    n.tops[prefix+'relu_conv1x1_1'] = L.ReLU(n.tops[prefix+'conv1x1_1'], in_place=True)

    n.tops[prefix+'conv1x1_2'] = L.Convolution(n.tops[prefix+'conv1x1_1'], kernel_size=1, num_output=128, weight_filler=dict(type='xavier'))
    n.tops[prefix+'relu_conv1x1_2'] = L.ReLU(n.tops[prefix+'conv1x1_2'], in_place=True)

    n.tops[prefix+'conv3x3_2'] = L.Convolution(n.tops[prefix+'conv1x1_1'], kernel_size=3, num_output=128, weight_filler=dict(type='xavier'))
    n.tops[prefix+'relu_conv3x3_2'] = L.ReLU(n.tops[prefix+'conv3x3_2'], in_place=True)

    n.tops[prefix+'concat'] = L.Concat(n.tops[prefix+'conv1x1_2'], n.tops[prefix+'conv3x3_2']) #is this right?

    next_bottom = prefix+'concat'
    return next_bottom

    #TODO: perhaps return the concat layer's name, so the next layer can use it as a 'bottom'

def FireNet(batch_size):
    n = NetSpec()
    n.data, n.label = L.DummyData(shape=[dict(dim=[batch_size, 1, 28, 28]),
                                         dict(dim=[batch_size, 1, 1, 1])],
                                  transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=7, num_output=96, stride=2, weight_filler=dict(type='xavier'))

    #TODO: loop over creation of FireNet modules
    curr_bottom = 'conv1'
    curr_bottom = FireNet_module(n, curr_bottom, 2) #should create Fire2/...
    curr_bottom = FireNet_module(n, curr_bottom, 3) #should create Fire3/...

    return n.to_proto()

#TODO: parameterize this more.
def lenet(batch_size):
    n = NetSpec()
    n.data, n.label = L.DummyData(shape=[dict(dim=[batch_size, 1, 28, 28]),
                                         dict(dim=[batch_size, 1, 1, 1])],
                                  transform_param=dict(scale=1./255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20,
        weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50,
        weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool2, num_output=500,
        weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=10,
        weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    n.tops['test/layer']=L.SoftmaxWithLoss(n.ip1, n.label)
    #embed()
    return n.to_proto()

if __name__ == "__main__":
    #net_proto = lenet(128)
    #save_prototxt(net_proto, 'lenet.prototxt')

    net_proto = FireNet(128)
    save_prototxt(net_proto, 'FireNet.prototxt')



