import os
from google.protobuf import text_format
from caffe.proto import caffe_pb2
from IPython import embed
from google.protobuf import text_format
#FIXME: importing stuff from util_FireNet is generating a bunch of warnings.
from gen_native_caffe_python_nets.util_FireNet import mkdir_p
from gen_native_caffe_python_nets.util_FireNet import save_prototxt
from gen_native_caffe_python_nets.util_FireNet import load_prototxt


def add_xavier(n_proto):
    for layer_ in n_proto.layer:
        if layer_.type == 'Convolution':
            layer_.convolution_param.weight_filler.type = 'xavier'
        elif layer_.type == 'InnerProduct':
            layer_.inner_product_param.weight_filler.type = 'xavier'

if __name__ == "__main__":
    inF = 'trainval_newFormat.prototxt'
    outF = 'trainval.prototxt'

    base_proto = load_prototxt(inF)

    add_xavier(base_proto)

    save_prototxt(base_proto, outF) 

