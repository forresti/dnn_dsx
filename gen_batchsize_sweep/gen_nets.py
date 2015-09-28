import caffe_pb2
from google.protobuf import text_format
import sys
import os
from pprint import pprint
from IPython import embed
from optparse import OptionParser
from collections import OrderedDict
from copy import deepcopy
from NetCreator import NetCreator 
from NetCreator import parse_options
from NiN_barebones import get_barebones_net

def mkdir_p(path):
    if not os.access(path, os.F_OK):
        os.mkdir(path)

#out_net is passed by ref
def customize_net(out_net, batch_size):
    #CUSTOMIZE THIS NET FOR PARAM SWEEP.
    if phase == 'trainval':
        out_net.layer[0].data_param.batch_size = batch_size

'''
def customize_solver(max_iter, stepsize):

'''


#TODO: SOLVER

if __name__ == '__main__':
    barebones_net = get_barebones_net()
    options = parse_options()
    phase = options['phase']

    netCreator = NetCreator()
    out_net = netCreator.create(barebones_net, phase)

    #TODO: loop over these:
    batch_size = 32
    LR_mult = float(2)
    out_dir = './NiN_batch%d_LR%0.1fx' %(batch_size, LR_mult)
    mkdir_p(out_dir)

    #propagate customizations to net
    customize_net(out_net, batch_size) #TODO: set LR
    out_net_file = out_dir + '/' + phase +'.prototxt'
    f = open(out_net_file, 'w')
    f.write(text_format.MessageToString(out_net))
    f.flush()
    f.close()


    #TODO: solver

