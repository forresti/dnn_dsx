
#ln -s caffe/python/caffe/proto/caffe_pb2.py ./

import caffe_pb2 # assume we created a protobuf type called "addressbook_pb2.py"
from google.protobuf import text_format
import sys
from random import seed
from random import choice
from random import uniform
from pprint import pprint
from IPython import embed


#@param net = NetParameter (loaded from prototxt file)
def getLayerByName(net, layerName):
    for L in net.layers:
        if L.name == layerName:
            return L

class NetMutator:

    def __init__(self):
        self.ranges = dict()
        self.ranges['conv'] = dict()
        self.ranges['pool'] = dict()

        self.ranges['conv']['num_output'] = [64, 128, 256, 512]
        self.ranges['conv']['kernel_size'] = [1,2,3,4,5,7,9,11]
        self.ranges['conv']['stride'] = self.ranges['conv']['kernel_size'] #and, when selecting stride, make sure it's <= kernel_size.

        self.ranges['pool']['type'] = [0, 1] #['MAX', 'AVE']
        self.ranges['pool']['kernel_size'] = [1,2,3,4,5,7,9,11]
        self.ranges['pool']['stride'] = self.ranges['pool']['kernel_size'] 

        #roughly 24 hyperparams that we play with in VGG-F. on average, mutate 2 params per net.
        self.mutation_prob = 1.0 / 12.0 

    #@param hp = hyperparam to modify (e.g. 'stride')
    # this updates 'net.' (passed by ref everywhere...)
    def mutate_layer(self, net, layerName, hp):
        #TODO
        L = getLayerByName(net, layerName)
        if L.HasField('pooling_param'):
            param_ = 'pooling_param'
            layerType = 'pool'
        elif L.HasField('convolution_param'):
            param_ = 'convolution_param'
            layerType = 'conv'
        else:
            print "ERROR: shouldn't mutate this layer type."
            sys.exit(1)

        hp_old_value = getattr(getattr(L, param_), hp) #e.g. net.layers[0].convolution_param.stride 

        if hp != 'stride':
            my_range = [v for v in self.ranges[layerType][hp] if v != hp_old_value] #don't mutate back to prev value. (TODO: test this)
            hp_new_value = choice(my_range)
            setattr(getattr(L, param_), hp, hp_new_value) # "net.layers[0].convolution_param.stride = hp_new_value"

        #else: #hp == 'stride'
        #TODO        


    #@param net = trainval.prototxt loaded into NetParameter structure
    #@return new_net = mutated net, which you can write to prototxt file
    #@return change_list = list of layers/hyperparams changed
    def mutate_layers(self, net):
        for L in net.layers:
            if L.HasField('pooling_param'):
                layerType = 'pool'
            #elif hasattr(L, 'convolution_param'): 
            elif L.HasField('convolution_param'):
                layerType = 'conv'
            else: continue #only mutating conv & pool layers.

            #print layerType
            for hp in self.ranges[layerType].keys():
                #print hp
                do_mutation = uniform(0,1) < self.mutation_prob #true for [0 to 1/12], else false.
                if do_mutation:
                    self.mutate_layer(net, L.name, hp) #pointer passing?

#TODO: make the following into a function that takes path to a net to mutate.
if __name__ == '__main__':
    seed(1) #TODO: take from cmd-line

    net = caffe_pb2.NetParameter()
    net_file = "/media/big_disk/installers_old/caffe-bvlc-master/models/bvlc_alexnet/deploy.prototxt"
    net_str = open(net_file).read()
    text_format.Merge(net_str, net) #net = load net from str to protobuf

    mutator = NetMutator()
    mutator.mutate_layers(net)
    embed()
