import sys
from random import seed
from random import choice
import random_net_defs #generate prototxt strings
from pprint import pprint
#Forrest's random CNN generator

#TODO: find a relatively elegant way to jam this data into a prototxt.
class RandomDNN:

    def __init__(self):
        self.min_kernelsize=1
        self.max_kernelsize=7
        self.min_num_output=20
        self.max_num_output=200
        self.min_lrn_dim=3
        self.max_lrn_dim=11
        self.max_layers=20

    #TODO: add 'group' parameter?
    def convLayer(self, layerIdx, prevLayerStr):
        myName = "layer" + str(layerIdx) + "_conv"
        kernelsize = choice(xrange(self.min_kernelsize, self.max_kernelsize+1))    
        num_output = choice(xrange(self.min_num_output, self.max_num_output+1))
        stride = choice(xrange(1, kernelsize+1)) #no point striding beyond the kernelsize, right?
        bottom = prevLayerStr
        top = myName

        #print myName, ' kernelsize = ', kernelsize, ' num_output = ', num_output, ' stride = ', stride
        retStr = random_net_defs.convLayerStr(myName, bottom, top, num_output, kernelsize, stride) 
        print retStr
        return {'name':myName, 'prototxt':retStr}

    def poolLayer(self, layerIdx, prevLayerStr):
        myName = "layer" + str(layerIdx) + "_pool"
        kernelsize = choice(xrange(self.min_kernelsize, self.max_kernelsize+1))
        stride = choice(xrange(1, kernelsize+1))
        poolType = choice(['AVE', 'MAX'])
        bottom = prevLayerStr
        top = myName

        #print myName, ' kernelsize = ', kernelsize, ' stride = ', stride, ' poolType = ', poolType
        retStr = random_net_defs.poolLayerStr(myName, bottom, top, kernelsize, stride, poolType)
        print retStr
        return {'name':myName, 'prototxt':retStr}


    def reluLayer(self, layerIdx, prevLayerStr):
        myName = "layer" + str(layerIdx) + "_relu"
        bottom = prevLayerStr
        top = myName #not bothering with in-place 

        #print myName
        retStr = random_net_defs.reluLayerStr(myName, bottom, top)
        print retStr
        return {'name':myName, 'prototxt':retStr}

    #alexnet, caffenet, GoogLeNet: all lrn layers have local_size=5
    def lrnLayer(self, layerIdx, prevLayerStr):
        myName = "layer" + str(layerIdx) + "_lrn" #called 'norm1', 'norm2', etc in most prototxts
        local_size = choice(xrange(self.min_lrn_dim, self.max_lrn_dim+2, 2)) #LRN supports odd values only
        bottom = prevLayerStr
        top = myName #not bothering with in-place 

        #print myName, ' local_size = ', local_size
        retStr = random_net_defs.lrnLayerStr(myName, bottom, top, local_size)
        print retStr
        return {'name':myName, 'prototxt':retStr}

    #CNN composition grammar
    def chooseNextLayer(self, prevLayerType):
        if prevLayerType=='conv':
            return 'relu'
        if prevLayerType=='data': #no point downsampling the input pixels (could do that at preprocessing time)
            return 'conv'
        else:
            return choice(['conv', 'lrn', 'pool'])
        #TODO: perhaps avoid 2 layers of the same type in a row.

#TODO: move this inside the RandomDNN class?
def gen_DNN():
    net = RandomDNN()
    data_layer_str = random_net_defs.dataLayerStr_deploy(256, 3, 227, 227)
    print data_layer_str

    prev_layer_type='data'
    prev_layer_name='data_layer'
    for i in xrange(0, 20):
        curr_layer_type = net.chooseNextLayer(prev_layer_type)
        if curr_layer_type == 'conv':
            curr_layer_dict = net.convLayer(i, prev_layer_name)
        if curr_layer_type == 'pool':
            curr_layer_dict = net.poolLayer(i, prev_layer_name)
        if curr_layer_type == 'relu':
            curr_layer_dict = net.reluLayer(i, prev_layer_name)
        if curr_layer_type == 'lrn':
            curr_layer_dict = net.lrnLayer(i, prev_layer_name)

        prev_layer_name = curr_layer_dict['name']
        prev_layer_type = curr_layer_type


if __name__ == "__main__":
    #TODO: make a class out of this.
    #seed(1)
    seed(sys.argv[1])
    #let's find boiler-plate places to put: DROPOUT, ACCURACY, SOFTMAX_LOSS
    
    #TODO: consider randomization within a layer, such as selecting MAX or AVG.

    net = RandomDNN()
    #net.convLayer(1, 'data')

    gen_DNN()

    '''
    prev_layer='data'
    for i in xrange(0, 20):
        curr_layer = net.chooseNextLayer(prev_layer)
        prev_layer = curr_layer
        print prev_layer
    '''
