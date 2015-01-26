
#sample usage: python mutate_net.py --model=VGG_F/trainval.prototxt
#              python mutate_net.py --model=VGG_F/deploy.prototxt --model_out=deploy_new.prototxt --seed=8
#ln -s caffe/python/caffe/proto/caffe_pb2.py ./
#this file is ONLY for mutating nets. crossover happens elsewhere (TODO: implement crossover.)

import caffe_pb2 # assume we created a protobuf type called "addressbook_pb2.py"
from google.protobuf import text_format
import sys
from random import seed
from random import choice
from random import uniform
from pprint import pprint
from IPython import embed
from optparse import OptionParser

#@param net = NetParameter (loaded from prototxt file)
def getLayerByName(net, layerName):
    for L in net.layers:
        if L.name == layerName:
            return L

class NetMutator:

    def __init__(self):
        self.ranges = dict()
        #TODO: make this an ordered dict so that 'kernel_size' comes before 'stride'
        self.ranges['conv'] = dict()
        self.ranges['pool'] = dict()

        self.ranges['conv']['num_output'] = [64, 128, 256, 512]
        self.ranges['conv']['kernel_size'] = [1,2,3,4,5,7,9,11]
        #self.ranges['conv']['stride'] = self.ranges['conv']['kernel_size'] #and, when selecting stride, make sure it's <= kernel_size.
        self.ranges['conv']['stride'] = [1,2,3,4]

        self.ranges['pool']['pool'] = [0, 1] #['MAX', 'AVE']
        self.ranges['pool']['kernel_size'] = [1,2,3,4,5,7,9,11]
        #self.ranges['pool']['stride'] = self.ranges['pool']['kernel_size'] 
        self.ranges['pool']['stride'] = [1,2,3,4]

        #roughly 24 hyperparams that we play with in VGG-F. on average, mutate 2 params per net.
        self.mutation_prob = 1.0 / 12.0 

    #@param hp = hyperparam to modify (e.g. 'stride')
    # this updates 'net.' (passed by ref everywhere...)
    def mutate_layer(self, net, layerName, hp):
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

        #choose new value for hyperparam.
        my_range = [v for v in self.ranges[layerType][hp] if v != hp_old_value] #don't mutate back to prev value.
        if hp == 'stride':
            my_kernel_size = getattr(getattr(L, param_), 'kernel_size')
            my_range = [v for v in my_range if v <= my_kernel_size] #ensure stride <= kernel_size
        hp_new_value = choice(my_range)

        #save new hp value to net 
        setattr(getattr(L, param_), hp, hp_new_value) # "net.layers[0].convolution_param.stride = hp_new_value"
        print 'updated %s: %s = %d' %(L.name, hp, hp_new_value)

        #if kernel_size is now larger than stride, decrement stride until stride <= kernel_size
        if hp == 'kernel_size' and hp_new_value < getattr(getattr(L, param_), 'stride'):
            stride_range = [v for v in self.ranges[layerType]['stride'] if v <= hp_new_value] #strides that are less than new kernel_size
            new_stride = stride_range[-1]
            setattr(getattr(L, param_), 'stride', new_stride) 
            print 'updated %s: %s = %d' %(L.name, 'stride', new_stride)

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

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--seed', '-s', type="int", help="OPTIONAL. seed for randomization (default: none)")
    parser.add_option('--model', type="string", help="REQUIRED. e.g. ...trainval.prototxt. you may want to use the result of crossover for this.")
    parser.add_option('--model_out', type="string", help="OPTIONAL. where to save the mutated output prototxt (default: ./trainval_mutated.prototxt")
    (options, args) = parser.parse_args()

    if options.seed is not None:
        seed(options.seed)

    if options.model is not None:
        net_file = options.model
    else:
        print "ERROR. --model is a mandatory flag"
        sys.exit(1)

    net = caffe_pb2.NetParameter()
    #net_file = "/media/big_disk/installers_old/caffe-bvlc-master/models/bvlc_alexnet/train_val.prototxt"
    net_str = open(net_file).read()
    text_format.Merge(net_str, net) #net = load net from str to protobuf

    mutator = NetMutator()
    mutator.mutate_layers(net)
    #embed()

    out_net_file = './trainval_mutated.prototxt'
    if options.model_out is not None:
        out_net_file = options.model_out

    f = open(out_net_file, 'w')
    f.write(text_format.MessageToString(net))
    f.flush()
    f.close()

