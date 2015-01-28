
import random_net_defs
import sys

#@param dnnDir = place where this DNN's trainval.prototxt is located (output data goes here, too)
def solver_generator(dnnDir):
    net_file = dnnDir + '/trainval.prototxt'
    snapshot_prefix = dnnDir + "/caffe_train"
    retStr = random_net_defs.solverStr(net_file, snapshot_prefix)
    return retStr

if __name__ == "__main__":
    dnnDir = sys.argv[1] #TODO: use flags instead?
    solverStr = solver_generator(dnnDir)
    print solverStr

