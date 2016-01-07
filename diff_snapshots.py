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
    #parser.add_argument('--prototxt', '-p', help="e.g. trainval.prototxt net definition", required=True)

    args = parser.parse_args()
    return args

#@param snapshot = path/to/*.caffemodel
def load_caffemodel(snapshot):
    net=caffe_pb2.NetParameter()
    net_input = open(snapshot, 'rb').read()
    net.ParseFromString(net_input)
    return net

#@param blob = RepeatedScalarFieldContainer ... i.e. blob, as represented in protobuf
def copy_RepeatedScalarFieldContainer_to_numpy(blob):
    out=np.zeros(len(blob), dtype=float)
    #TODO: perhaps speed this up if it's a bottleneck
    for i in xrange(0, len(blob)):
        out[i] = blob[i]

    return out

#TODO: delete... just for my own learning.
def toy_example_load_snapshot():
    snapshot1 = 'diff_snapshots_data/nin_imagenet_train_iter_2000.caffemodel'
    net1 = load_caffemodel(snapshot1)

#@param np1, np2 = two 1D numpy vectors of the same size
def my_diff(np1, np2):
    np1 = np.absolute(np1)
    np2 = np.absolute(np2)
    elemwise_diff = np2 - np1
    total_diff = sum(elemwise_diff)
    return total_diff

#@param s1, s2 = NetParameter objects.
def diff_snapshots(s1, s2):
    print " calculating sum of diffs as sum(|s1|-|s2|)"
    for L in xrange(0, len(s1.layer)):
        Ls1 = s1.layer[L]
        Ls2 = s2.layer[L]

        #TODO: perhaps just check whether the layer has learned parameters?
        if not( (Ls1.type == 'Convolution') or (Ls1.type == 'InnerProduct')):
            continue

        #FILTERS
        param_s1 = Ls1.blobs[0].data #google.protobuf.internal.containers.RepeatedScalarFieldContainer
        param_s2 = Ls2.blobs[0].data

        #convert to numpy array
        param_s1 = copy_RepeatedScalarFieldContainer_to_numpy(param_s1)
        param_s2 = copy_RepeatedScalarFieldContainer_to_numpy(param_s2)

        diff = my_diff(param_s1, param_s2)
        print "  %s sum of diff: %f" %(Ls1.name, diff)

        #BIAS
        #param_s1 = Ls1.blob[1]

        

if __name__ == "__main__":

    #toy_example_load_snapshot()
    args = parse_arguments()
    s1 = load_caffemodel(args.snapshot1)
    s2 = load_caffemodel(args.snapshot2)

    #TODO: test that all s1 and s2 have the same DNN architecture.
    diff_snapshots(s1,s2)

    #print args
    #print args.prototxt



