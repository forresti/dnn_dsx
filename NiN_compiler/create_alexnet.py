import caffe_pb2
from google.protobuf import text_format
import sys
from pprint import pprint
from IPython import embed
from optparse import OptionParser
from collections import OrderedDict
from copy import deepcopy
from NetCreator import NetCreator 
from NetCreator import parse_options

def get_barebones_net():

    barebones_net = OrderedDict() 
    barebones_net['conv1'] = {'type': "Convolution", 'convolution_param':{'num_output':96, 'kernel_size':11, 'stride':4}} 
    barebones_net['relu_conv1'] = {'type': "ReLU"}

    barebones_net['pool0'] = {'type': "Pooling"} #assume defaults: MAX, ksize=3, stride=2

    barebones_net['conv2'] = {'type': "Convolution", 'convolution_param':{'num_output':256, 'kernel_size':5, 'stride':1, 'pad':2}}
    barebones_net['relu_conv2'] = {'type': "ReLU"}

    barebones_net['pool1'] = {'type': "Pooling"} #assume defaults: MAX, ksize=3, stride=2

    barebones_net['conv3'] = {'type': "Convolution", 'convolution_param':{'num_output':384, 'kernel_size':3, 'stride':1, 'pad':1}}
    barebones_net['relu_conv3'] = {'type': "ReLU"}

    barebones_net['conv4'] = {'type': "Convolution", 'convolution_param':{'num_output':384, 'kernel_size':3, 'stride':1, 'pad':1}}
    barebones_net['relu_conv4'] = {'type': "ReLU"}

    barebones_net['conv5'] = {'type': "Convolution", 'convolution_param':{'num_output':256, 'kernel_size':3, 'stride':1, 'pad':1}}
    barebones_net['relu_conv5'] = {'type': "ReLU"}

    barebones_net['pool2'] = {'type': "Pooling"}

  #fc layers (as conv)
    barebones_net['fc6'] = {'type': "Convolution", 'convolution_param':{'num_output':4096, 'kernel_size':6, 'stride':1}} 
    barebones_net['relu_fc6'] = {'type': "ReLU"}
    barebones_net['dropout_fc6'] = {'type': "Dropout"}

    barebones_net['fc7'] = {'type': "Convolution", 'convolution_param':{'num_output':4096, 'kernel_size':1, 'stride':1}} 
    barebones_net['relu_fc7'] = {'type': "ReLU"}
    barebones_net['dropout_fc7'] = {'type': "Dropout"}

    barebones_net['fc8'] = {'type': "Convolution", 'convolution_param':{'num_output':1000, 'kernel_size':1, 'stride':1}}
    barebones_net['relu_fc8'] = {'type': "ReLU"}

    #put 'name' into each dict
    for k in barebones_net.keys():
        barebones_net[k]['name']=k

    return barebones_net

if __name__ == '__main__':
    barebones_net = get_barebones_net()
    options = parse_options()
    phase = options['phase']

    netCreator = NetCreator()
    out_net = netCreator.create(barebones_net, phase)
    out_net_file = './' + phase +'_alexnet.prototxt'
    f = open(out_net_file, 'w')
    f.write(text_format.MessageToString(out_net))
    f.flush()
    f.close()
