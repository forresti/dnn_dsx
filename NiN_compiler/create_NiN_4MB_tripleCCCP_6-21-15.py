import caffe_pb2 # assume we created a protobuf type called "addressbook_pb2.py"
from google.protobuf import text_format
import sys
from pprint import pprint
from IPython import embed
from optparse import OptionParser
from collections import OrderedDict
from copy import deepcopy
from NetCreator import NetCreator 

if __name__ == '__main__':

    barebones_net = OrderedDict() 
    barebones_net['conv1'] = {'type': "Convolution", 'convolution_param':{'num_output':64, 'kernel_size':11, 'stride':4}} 
    barebones_net['relu_conv1'] = {'type': "ReLU"}

    barebones_net['cccp1'] = {'type': "Convolution", 'convolution_param':{'num_output':96, 'kernel_size':1, 'stride':1}} 
    barebones_net['relu_cccp1'] = {'type': "ReLU"}

    barebones_net['cccp2'] = {'type': "Convolution", 'convolution_param':{'num_output':96, 'kernel_size':1, 'stride':1}} 
    barebones_net['relu_cccp2'] = {'type': "ReLU"}

    barebones_net['cccp3'] = {'type': "Convolution", 'convolution_param':{'num_output':64, 'kernel_size':1, 'stride':1}}
    barebones_net['relu_cccp3'] = {'type': "ReLU"}

    barebones_net['pool0'] = {'type': "Pooling"} #assume defaults: MAX, ksize=3, stride=2

#conv2 - cccp4
    barebones_net['conv2'] = {'type': "Convolution", 'convolution_param':{'num_output':96, 'kernel_size':5, 'stride':2, 'pad':2}}
    barebones_net['relu_conv2'] = {'type': "ReLU"}

    barebones_net['cccp4'] = {'type': "Convolution", 'convolution_param':{'num_output':128, 'kernel_size':1, 'stride':1}}
    barebones_net['relu_cccp4'] = {'type': "ReLU"}

    barebones_net['cccp5'] = {'type': "Convolution", 'convolution_param':{'num_output':128, 'kernel_size':1, 'stride':1}}
    barebones_net['relu_cccp5'] = {'type': "ReLU"}

    barebones_net['cccp6'] = {'type': "Convolution", 'convolution_param':{'num_output':96, 'kernel_size':1, 'stride':1}}
    barebones_net['relu_cccp6'] = {'type': "ReLU"}

    barebones_net['pool1'] = {'type': "Pooling"} #assume defaults: MAX, ksize=3, stride=2

#conv3 - cccp6
    barebones_net['conv3'] = {'type': "Convolution", 'convolution_param':{'num_output':128, 'kernel_size':3, 'stride':1, 'pad':1}} #TODO: oblong filters.
    barebones_net['relu_conv3'] = {'type': "ReLU"}

    barebones_net['cccp7'] = {'type': "Convolution", 'convolution_param':{'num_output':256, 'kernel_size':1, 'stride':1}}
    barebones_net['relu_cccp7'] = {'type': "ReLU"}

    barebones_net['cccp8'] = {'type': "Convolution", 'convolution_param':{'num_output':128, 'kernel_size':1, 'stride':1}}
    barebones_net['relu_cccp8'] = {'type': "ReLU"}

    barebones_net['pool2'] = {'type': "Pooling"} #assume defaults: MAX, ksize=3, stride=2

#conv4 - cccp8
    barebones_net['conv4'] = {'type': "Convolution", 'convolution_param':{'num_output':192, 'kernel_size':3, 'stride':1, 'pad':1}}
    barebones_net['relu_conv4'] = {'type': "ReLU"}

    barebones_net['cccp9'] = {'type': "Convolution", 'convolution_param':{'num_output':384, 'kernel_size':1, 'stride':1}}
    barebones_net['relu_cccp9'] = {'type': "ReLU"}

    barebones_net['cccp10'] = {'type': "Convolution", 'convolution_param':{'num_output':1000, 'kernel_size':1, 'stride':1}}
    barebones_net['relu_cccp10'] = {'type': "ReLU"}

    barebones_net['pool3'] = {'type': "Pooling", 'pooling_param':{'pool':1}} #MAX=0, AVE=1 ... the enum appears as 'AVE' when written to disk.

    #put 'name' into each dict
    for k in barebones_net.keys():
        barebones_net[k]['name']=k

    netCreator = NetCreator()
    out_net = netCreator.create(barebones_net)
    out_net_file = './base_NiN_4MB_tripleCCCP_6-21-15.prototxt'
    f = open(out_net_file, 'w')
    f.write(text_format.MessageToString(out_net))
    f.flush()
    f.close()
