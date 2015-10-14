import caffe_pb2
from google.protobuf import text_format
import sys
import os
import math
from pprint import pprint
from IPython import embed
from optparse import OptionParser
from collections import OrderedDict
from copy import deepcopy
from shutil import copyfile
#from NetCreator import NetCreator 
#from NetCreator import parse_options
#from NetCreator import getLayersByName
#from NiN_conv1_st2_conv2_st2 import get_barebones_net

#round up to nearest 1000
def roundup(x):
    return int(math.ceil(x / 1000.0)) * 1000

def mkdir_p(path):
    if not os.access(path, os.F_OK):
        os.mkdir(path)

#my 'default' at batch 256, 46.9 epochs
#max_iter = 220k
#stepsize = 100k // max_iter * (10.0 / 22.0)

#@return {base_lr, stepsize, max_iter}
def derive_training_params(batch_size, num_epochs, LR_mult):
    #default settings... (could make these input args)
    LR_at_batch_256 = 0.01
    epoch_length = 1200000 #imagenet 1.2M images

    #derived settings 
    base_lr = LR_at_batch_256 * (batch_size / 256.0) * LR_mult
    max_iter = float(epoch_length) * num_epochs / batch_size
    max_iter = roundup(max_iter)

    stepsize = max_iter * (10.0 / 22.0) 
    stepsize = roundup(stepsize)

    n_gpu = math.ceil(batch_size/64.0) #number of GPUs to use for training... batch=128 -> 2 gpu. batch=256 -> 4 GPUs.
    snapshot = (32.0/batch_size) * 4000 * n_gpu #batch=32 -> snapshot=4000. batch=64 -> snapshot=2000. and, snapshot less often if we have more GPUs.
    test_interval = (256.0/batch_size) * 1000 * n_gpu #batch=256 -> test_iter=1000. and, test less often if we have more GPUs.
    display = max( 20, (256.0/batch_size) * 20) #don't print too often...


    return {'base_lr':float(base_lr), 'max_iter':int(max_iter), 'stepsize':int(stepsize), 'snapshot':int(snapshot), 'test_interval':int(test_interval), 'n_gpu':int(n_gpu), 'display':int(display)}

'''
#out_net is passed by ref
def customize_net(out_net, batch_size):
    #CUSTOMIZE THIS NET FOR PARAM SWEEP.
    if phase == 'trainval':
        out_net.layer[0].data_param.batch_size = batch_size

    #enable globalpooling in pool4. (TODO: move this to NetCreator...)
    pool4=getLayersByName(out_net, 'pool4')[0]
    p=pool4.pooling_param
    p.ClearField('stride')
    p.ClearField('kernel_size')
    p.global_pooling=1
'''

def load_solver():
    solver_file = 'base_solver.prototxt'
    solver_str = open(solver_file).read()
    solver = caffe_pb2.SolverParameter()
    text_format.Merge(solver_str, solver)
    return solver

#solver is passed by ref
#hparams = dictionary of derived hyperparameters
def customize_solver(solver, hparams): #max_iter, stepsize, base_lr, snapshot, test_interval):
    #solver.max_iter = hparams['max_iter']
    #solver.stepsize = hparams['stepsize']
    solver.base_lr = hparams['base_lr']
    #solver.snapshot = hparams['snapshot']
    #solver.test_interval = hparams['test_interval']
    #solver.display = hparams['display']

def save_prototxt(protobuf, out_fname):
    f = open(out_fname, 'w')
    f.write(text_format.MessageToString(protobuf))
    f.flush()
    f.close()

#def one_DSE_net(phase, batch_size, LR_mult, num_epochs):
def one_DSE_net(LR_mult):
    #p = derive_training_params(batch_size, num_epochs, LR_mult)
    p = dict()
    p['base_lr'] = 0.32
    p['n_gpu'] = 128
    num_epochs=64
    batch_size=1024

    '''
    #TODO: print "set: 30 epochs, batch=32, 2xLR" ; "got: LR, step, max_iter"
    print '  user selected: num_epochs=%f, batch_size=%d, LR_mult=%f' %(num_epochs, batch_size, LR_mult)
    #print '    got: max_iter=%d, stepsize=%d, base_lr=%f, snapshot=%d, test_interval=%d' %(p['max_iter'], p['stepsize'], p['base_lr'], p['snapshot'], p['test_interval'])
    print "  got:", p
    '''

    out_parent_dir = './nets_batchsize_sweep'
    mkdir_p(out_parent_dir)
    out_dir = out_parent_dir + '/googlenet_epochs%0.1f_batch%04d_LR%0.3fx' %(num_epochs, batch_size, LR_mult)
    mkdir_p(out_dir)

    #boilerplate
    '''
    barebones_net = get_barebones_net()
    netCreator = NetCreator()
    out_net = netCreator.create(barebones_net, phase)

    #propagate customizations to net
    customize_net(out_net, batch_size)
    out_net_file = out_dir + '/' + phase +'.prototxt'
    save_prototxt(out_net, out_net_file)
    '''
    copyfile('base_trainval.prototxt', out_dir + '/trainval.prototxt')


    #update solver 
    solver = load_solver() #default solver
    customize_solver(solver, p) #p['max_iter'], p['stepsize'], p['base_lr'], p['snapshot'], p['test_interval'])
    out_solver_file = out_dir + '/solver.prototxt'
    save_prototxt(solver, out_solver_file)

    #file with one number: how many GPUs to use
    out_gpu_file = out_dir + '/n_gpu.txt'
    f = open(out_gpu_file, 'w')
    f.write( str(p['n_gpu']) )
    f.close()

if __name__ == '__main__':
    '''
    options = parse_options()
    phase = options['phase']

    num_epochs = 46.9
    batch_sizes = x=[2**x for x in xrange(5, 12)] #32 to 2048
    #batch_sizes = x=[2**x for x in xrange(5, 8)] #32 to 128

    for LR_mult in [0.5, 1.0, 2.0]:
        for batch_size in batch_sizes:
            one_DSE_net(phase, batch_size, LR_mult, num_epochs)
    '''

    for LR_mult in [0.125, 0.25, 0.5, 1.0]:
        one_DSE_net(LR_mult)

