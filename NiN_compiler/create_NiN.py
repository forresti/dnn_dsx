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
    #barebones_net['data'] = 'trainval'
    barebones_net['conv1'] = {'name': "conv1", 'type': "Convolution", 'convolution_param':{'num_output':96, 'kernel_size':11, 'stride':2}} #actually want stride=4 for typical NiN
    barebones_net['relu1'] = {'name': "relu_conv1", 'type': "ReLU"}
    #barebones_net['accuracy'] = None
    #barebones_net[''] = None

    netCreator = NetCreator()
    out_net = netCreator.create(barebones_net)

    out_net_file = './base_NiN.prototxt'
    #if options.model_out is not None:
    #    out_net_file = options.model_out

    f = open(out_net_file, 'w')
    f.write(text_format.MessageToString(out_net))
    f.flush()
    f.close()
