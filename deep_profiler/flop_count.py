from IPython import embed
from google.protobuf import text_format
import numpy as np
from optparse import OptionParser
from argparse import ArgumentParser
from caffe.proto import caffe_pb2
import caffe
#currently: not FLOPS/s, just '# FLOPS to compute;

'''
e.g.
t=/Users/forrest/computerVision/dnn_dsx/gen_native_caffe_python_nets/nets_old/FireNet_8_fireLayers_base_r_64_64_incr_r_64_64_CEratio_0.125_freq_2/trainval.prototxt
python ./flop_count.py -t $t
'''

def load_trainval_prototxt(fname):
    return_net = caffe_pb2.NetParameter()
    net_str = open(fname).read()
    text_format.Merge(net_str, return_net)
    return return_net

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--trainval', '-t', help="path to trainval.prototxt", required=True)

    args = parser.parse_args()
    return args

#TODO: decide whether to my own calculation of activation plane sizes, or to initialize Caffe and see what it gets...

#def get_output_sizes(net):
#    for L in net.blobs.keys():
#       net.blobs[L].data.size

if __name__ == '__main__':
    args = parse_arguments()
    #trainval = load_trainval_prototxt(args.trainval)
    net = caffe.Net(args.trainval, caffe.TEST)    
    #arghhhhh, it wants the LMDB path to exist, even though I'm not using it here. arghhhhh.

    #TODO:
    # (should probably get pycaffe to do the following for me...) 
    #1. calculate # of input channels to each layer
    #2. calculate activation plane sizes
    embed()
    #next. do some parsing and FLOP counting...

