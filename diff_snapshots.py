from IPython import embed
import numpy as np
from optparse import OptionParser
from argparse import ArgumentParser
from caffe.proto import caffe_pb2
#from caffe import Net

def parse_arguments():
    #parser = OptionParser()
    parser = ArgumentParser()
    parser.add_argument('--snapshot1', '-s1', help="snapshot #1", required=True)
    parser.add_argument('--snapshot2', '-s2', help="snapshot #2", required=True)
    parser.add_argument('--prototxt', '-p', help="e.g. trainval.prototxt net definition", required=True)

    args = parser.parse_args()
    return args

#@param snapshot = path/to/*.caffemodel
def load_caffemodel(snapshot):
    net=caffe_pb2.NetParameter()
    net_input = open(snapshot, 'rb').read()
    net.ParseFromString(net_input)
    return net

#TODO: delete... just for my own learning.
def toy_example_load_snapshot():
    snapshot1 = 'diff_snapshots_data/nin_imagenet_train_iter_2000.caffemodel'
    net1 = load_caffemodel(snapshot1)

if __name__ == "__main__":

    toy_example_load_snapshot()
    #args = parse_arguments()

    #print args
    #print args.prototxt



