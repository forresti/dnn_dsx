
#this is tested on a20.mill (doesn't currently work on a19 due to prototxt install issue.)

import caffe
#from caffe.io import load_image
from caffe import Classifier
from IPython import embed
import numpy as np
import time
import pickle
from glob import glob
import sys

#'Net' likes being accessed from a child class.
# modified from classifier.py
class NetExtender(caffe.Net):
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 gpu=False, mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):

        caffe.Net.__init__(self, model_file, pretrained_file)

def compute_throughput(prototxt, caffemodel):
    net_ext = NetExtender(prototxt, caffemodel)

    if 'data' in net_ext.blobs.keys(): #in most nets
        data_count = net_ext.blobs['data'].count
    else: #in my auto-generated nets
        data_count = net_ext.blobs['data_layer'].count

    count_per_layer = dict()
    throughput_per_layer = dict()

    #for layerName in net_ext.params.keys():
    for layerName in net_ext.blobs.keys():
        count = net_ext.blobs[layerName].count
        count_per_layer[layerName] = count
        throughput_per_layer[layerName] =  float(count) / float(data_count)

        print 'throughput of ',layerName,' = ',throughput_per_layer[layerName]

'''
sed -i 's/\/lustre\/atlas\/scratch\/forresti\/csc103\/dnn_exploration\/caffe-bvlc-master\/data\/ilsvrc12\/imagenet_mean.binaryproto/\/home\/eecs\/forresti\/caffe_depthMax_and_hog\/data\/ilsvrc12\/imagenet_mean.binaryproto/g' 1/trainval.prototxt

sed -i 's/\/lustre\/atlas\/scratch\/forresti\/csc103\/dnn_exploration\/ilsvrc2012_train_256x256_lmdb/\/scratch\/forresti\/ilsvrc2012\/ilsvrc2012_train_256x256_lmdb/g' 1/trainval.prototxt

sed -i 's/\/lustre\/atlas\/scratch\/forresti\/csc103\/dnn_exploration\/ilsvrc2012_val_256x256_lmdb/\/scratch\/forresti\/ilsvrc2012\/ilsvrc2012_val_256x256_lmdb/g' 1/trainval.prototxt
'''

if __name__ == "__main__":

    #do we really need the pretrained model?

    #prototxt = '/home/eecs/forresti/caffe_depthMax_and_hog/nets/caffenet_finetune_flickrlogos/trainval.prototxt'
    #caffemodel = '/home/eecs/forresti/caffe_depthMax_and_hog/nets/caffenet_finetune_flickrlogos/bvlc_reference_caffenet.caffemodel'
    #prototxt = '/home/eecs/forresti/caffe_depthMax_and_hog/models/VGG/VGG_ILSVRC_19_layers_deploy.prototxt'
    #caffemodel = '/home/eecs/forresti/caffe_depthMax_and_hog/models/VGG/VGG_ILSVRC_19_layers.caffemodel'
    #prototxt='/home/eecs/forresti/caffe_depthMax_and_hog/nets/99/trainval.prototxt'
    #caffemodel='/home/eecs/forresti/caffe_depthMax_and_hog/nets/99/caffe_train_iter_450000.caffemodel'
   
    #TODO: take basedir as input arg

    if len(sys.argv) == 2:
        baseDir = sys.argv[1]
    else:
        baseDir='/nscratch/forresti/dsx_backup/nets_backup_1-9-15/1'
    prototxt=baseDir+'/trainval.prototxt'
    caffemodel = glob(baseDir+'/*.caffemodel')[0]

    #prototxt='/nscratch/forresti/dsx_backup/nets_backup_1-9-15/1/trainval.prototxt'
    #caffemodel = glob('/nscratch/forresti/dsx_backup/nets_backup_1-9-15/1/*.caffemodel')[0] #doesn't matter which one we pick... just doing dimension book-keeping. 

    compute_throughput(prototxt, caffemodel)

