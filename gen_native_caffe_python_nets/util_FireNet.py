from math import floor
import os
from caffe import Net
from caffe import NetSpec
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from IPython import embed
from google.protobuf import text_format
from random import choice
from random import seed
from shutil import copyfile
import conf_firenet as conf

def mkdir_p(path):
    if not os.access(path, os.F_OK):
        os.mkdir(path)

def save_prototxt(protobuf, out_fname):
    f = open(out_fname, 'w')
    f.write(text_format.MessageToString(protobuf))
    f.flush()
    f.close()


def FireNet_pooling_layer(n, bottom, pool_spec, layer_idx):
    p = pool_spec
    next_bottom='pool'+str(layer_idx)
    n.tops[next_bottom] = L.Pooling(n.tops[bottom], kernel_size=p['kernel_size'],
                                                stride=p['stride'], pool=p['pool'])
    return next_bottom


#@param NetSpec n
def FireNet_data_layer(n, batch_size):
    #important: conf.test_lmdb
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=conf.test_lmdb, include=dict(phase=caffe_pb2.TEST),
                             transform_param=dict(crop_size=227, mean_value=[104, 117, 123]), ntop=2)

#get protobuf containing only a data layer
def get_train_data_layer(batch_size):
    #important: conf.train_lmdb
    n = NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=conf.train_lmdb, include=dict(phase=caffe_pb2.TRAIN),
                             transform_param=dict(crop_size=227, mean_value=[104, 117, 123]), ntop=2)
    return n.to_proto()

#@return pool_after_dict = ['conv1': {'stride': 2, 'pool': 0, 'kernel_size': 3}, ...]
#@return pool_after = [1,2,4]
def get_randomized_pooling_scheme(n_layers, n_poolings):
    pool_after = [1] #for now, always pool after conv1.
    remaining_poolings = n_poolings - 1

    #TODO: assert(n_poolings >= n_layers)
    #TODO: random seed

    choices = [x for x in xrange(2, n_layers+1)]
    while remaining_poolings > 0:
        choices_ind = [x for x in xrange(0, len(choices))]
        c_idx = choice(choices_ind)
        curr_pool = choices.pop(c_idx) #grab this choice AND remove it from the list of choices
        pool_after.append(curr_pool)
        remaining_poolings = remaining_poolings - 1
    pool_after = sorted(pool_after)

    pool_after_dict = dict()
    for p in pool_after:
        if p==1:
            pool_after_dict['conv1'] = regular_pool
        else:
            pool_after_dict['fire%d/concat'%p] = regular_pool

    return [pool_after_dict, pool_after]


