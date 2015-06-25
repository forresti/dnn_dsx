import caffe_pb2 # assume we created a protobuf type called "addressbook_pb2.py"
from google.protobuf import text_format
import sys
from pprint import pprint
from IPython import embed
from optparse import OptionParser
from collections import OrderedDict
from copy import deepcopy

#Designed for compatibility with current prototxt format as of 6/25/15

#@param net = NetParameter (loaded from prototxt file)
def getLayerByName(net, layerName):
    for L in net.layer:
        if L.name == layerName:
            return L

def getLayersByType(net, layerType):
    retList = []
    for L in net.layer:
        if L.type == layerType:
            retList.append(L)
    return retList

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
        if d['type'] == 'ReLU': #in-place buffer
            b.extend([curr_input_blob])
            t.extend([curr_input_blob])
            #setattr(pb, 'bottom', [curr_input_blob])
            #setattr(pb, 'top', curr_input_blob)

        else: #out of place buffer
            b.extend([curr_input_blob])
            t.extend([d['name']])
            curr_input_blob = d['name']

        #pb is updated in-place
        return curr_input_blob

    #@param barebones_net_dict = {{'type':'Convolution, 'name':'conv1'}, ...} -- anything not specified will go to default.
    # ASSUME: barebones_net_dict is an OrderedDict w/ layers in topological order
    # ASSUME: barebones_net_dict defines a linked-list topology (can support DAGs later)
    def create(self, barebones_net_dict):
        output_net = caffe_pb2.NetParameter() 
        curr_input_blob = 'data' #TODO: update this after each non-ReLU layer

        #for L in self.net_inherit.layer:
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
        return output_net 

