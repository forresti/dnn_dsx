import caffe_pb2
from google.protobuf import text_format
import sys
from pprint import pprint
from IPython import embed
from optparse import OptionParser
from collections import OrderedDict
from copy import deepcopy

#Designed for compatibility with current prototxt format as of 6/25/15

#@param net = NetParameter (loaded from prototxt file)
def getLayersByName(net, layerName):
    retList = []
    for L in net.layer:
        if L.name == layerName:
            retList.append(L)
    return retList

def getLayersByType(net, layerType):
    retList = []
    for L in net.layer:
        if L.type == layerType:
            retList.append(L)
    return retList

def parse_options():
    parser = OptionParser()
    parser.add_option('--phase', '-p', type="string", help="OPTIONAL: 'trainval' or 'deploy', default is 'deploy'")
    (options, args) = parser.parse_args()
    phase = 'deploy'
    if options.phase is not None:
        if options.phase == 'trainval':
            phase = 'trainval'
        elif options.phase == 'deploy':
            phase='deploy'
        else:
            print "unknown phase argument. defaulting to 'deploy'"

    return {'phase':phase}

#TODO: put this in NetCreator.py
# planned usage: "NiN_deep.py defines a custom net... and it uses NetCreator to generate the net prototxt with all the minutae in it"
class NetCreator:
    def __init__(self):
        #net_inherit = prototxt with some default layers (we use these as templates)
        self.net_inherit = caffe_pb2.NetParameter()
        net_file = 'base_template.prototxt'
        net_str = open(net_file).read()
        text_format.Merge(net_str, self.net_inherit) #net = load net from str to protobuf

    #@param d = dict of customizations (for one layer)
    #@param pb = protobuf to modify (for one layer)
    def mirror_dict_to_protobuf(self, d, pb):
        for k in d.keys():
            d_elem = d[k]
            pb_elem = getattr(pb, k) #TODO: error check for key missing from pb
            if isinstance(d_elem, dict): 
                self.mirror_dict_to_protobuf(d_elem, pb_elem)
            else: #base case... this gets assigned directly to pb
                setattr(pb, k, d_elem)
        #no return... pb is updated in-place.

    #TODO: {trainval, deploy}_{prefix, suffix}() ... for data loading & scoring info.
    #Usage: apply this to an empty NetParameter object
    def deploy_prefix(self, output_net):
        input_ = ['data']
        input_dim = [50, 3, 227, 227]
        output_net.input.extend(input_)
        output_net.input_dim.extend(input_dim)  

    #Usage: apply this to an empty NetParameter object
    def trainval_prefix(self, output_net):
        input_layers = getLayersByName(self.net_inherit, 'data') #includes train & test versions
        output_net.layer.extend(input_layers) 

    def deploy_suffix(self, output_net, curr_input_blob):
        softmax_layer = getLayersByName(self.net_inherit, 'softmax_deploy')[0]
        softmax_layer.bottom.extend([curr_input_blob])
        output_net.layer.extend([softmax_layer])

    #assume 'bottom' is empty in softmax_trainval template
    def trainval_suffix(self, output_net, curr_input_blob):
        softmax_layer = deepcopy( getLayersByName(self.net_inherit, 'softmax_trainval')[0] )
        softmax_layer.bottom.extend([curr_input_blob, 'label'])

        accuracy_layer = deepcopy( getLayersByName(self.net_inherit, 'accuracy_trainval')[0] )
        accuracy_layer.bottom.extend([curr_input_blob, 'label'])

        output_net.layer.extend([softmax_layer, accuracy_layer])

    #wire layer inputs/outputs together
    #param d, pb -> see mirror_dict_to_protobuf
    def blob_IO(self, d, pb, curr_input_blob):

        #clear existing bottom/top stuff (might present a problem for 'Accuracy' layer w/ multiple inputs)
        b = pb.bottom
        t = pb.top
        while len(b) > 0:
            del b[0]
        while len(t) > 0:
            del t[0] 

        #update bottom & top
        if d['type'] == 'ReLU' or d['type'] == 'Dropout': #in-place buffer
            b.extend([curr_input_blob])
            t.extend([curr_input_blob])

        else: #out of place buffer
            b.extend([curr_input_blob])
            t.extend([d['name']])
            curr_input_blob = d['name']

        #pb is updated in-place
        return curr_input_blob

    #@param barebones_net_dict = {{'type':'Convolution, 'name':'conv1'}, ...} -- anything not specified will go to default.
    # ASSUME: barebones_net_dict is an OrderedDict w/ layers in topological order
    # ASSUME: barebones_net_dict defines a linked-list topology (can support DAGs later)
    #@param phase = 'trainval' or 'deploy'
    def create(self, barebones_net_dict, phase):
        if (phase != 'trainval') and (phase != 'deploy'): #TODO: make sure this test works
            print "phase must be 'trainval' or 'deploy'"
            return None

        output_net = caffe_pb2.NetParameter() 
        curr_input_blob = 'data' #TODO: update this after each non-ReLU layer
        if phase == 'deploy':
            self.deploy_prefix(output_net) #TODO: if/else trainval/deploy
        else:
            self.trainval_prefix(output_net)

        for L in barebones_net_dict:
            layer_type = barebones_net_dict[L]['type'] #e.g. 'Convolution'
            my_template = getLayersByType(self.net_inherit, layer_type)[0] 
            
            #1. copy template
            my_layer = deepcopy(my_template)

            #2. modify template
            self.mirror_dict_to_protobuf(barebones_net_dict[L], my_layer) #customize my_layer based on barebones_net_dict
            curr_input_blob = self.blob_IO(barebones_net_dict[L], my_layer, curr_input_blob)

            #3. append modified template to output net
            output_net.layer.extend([my_layer])

        if phase == 'deploy':
            self.deploy_suffix(output_net, curr_input_blob)
        else:
            self.trainval_suffix(output_net, curr_input_blob)

        return output_net 

