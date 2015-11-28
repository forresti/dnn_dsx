from math import floor
from pprint import pprint

'''
  memory / stride tradeoffs.
  simplifying assumptions for conv layers:
    - ignore padding and filter height/width
    - this would be 'correct' if all conv filters were 1x1 with pad=0
  simplifying assumptions for pooling layers:
    - assume padding and filter resolution cancel out... 
    - in other words, we divide data plane height & width by stride after each pooling 
    - assume there's a final conv-1000 and global pooling layer at the end (which we ignore here)
'''

def choose_filters_per_layer():
    num_layers=10
    base = 64
    incr = 64
    filters_per_layer = []
    for i in xrange(0, num_layers):
        filters_per_layer.append(base + i*incr)

    return filters_per_layer

#figure out total memory used for data planes.
def compute_planes_size(plane_dims):
    total_planes_size = 0
    for k in plane_dims.keys():
        L = plane_dims[k]
        plane_size = L['activ_h'] * L['activ_w'] * L['chans_out'] * L['batch'] * 4 #4 bytes/float
        plane_size = float(plane_size) / 1048576.0 #bytes to MB
        plane_dims[k]['plane_size'] = plane_size
        total_planes_size = total_planes_size + plane_size 

    return total_planes_size

if __name__ == "__main__":

    s = dict() #settings

    #FireNet_default (not including beginning and ending conv layers)
    s['num_layers'] = 4
    s['batch'] = 1024
    s['input_h'] = 224/2 #factor in stride=2 for conv1
    s['input_w'] = 224/2 

    s['filters_per_layer'] = choose_filters_per_layer()
    s['pool_stride'] = 2 #static for all layers, for now. 
    #s['pool_after'] = [0,4,9]
    s['pool_after'] = [0,1,2]

    plane_dims=dict() #for all layers
    prev_layer = {'activ_w':s['input_w'], 'activ_h':s['input_h'], 'chans_out':3, 'batch':s['batch']}
    plane_dims['data'] = prev_layer

    for layer_idx in xrange(0, len(s['filters_per_layer'])):
        layer_name = "conv%d" %layer_idx
        curr_layer = dict()
        if layer_idx in s['pool_after']:
            curr_layer['activ_w'] = prev_layer['activ_w']/2
            curr_layer['activ_h'] = prev_layer['activ_h']/2
        else: #no pooling here.
            curr_layer['activ_w'] = prev_layer['activ_w']
            curr_layer['activ_h'] = prev_layer['activ_h']

        curr_layer['chans_out'] = s['filters_per_layer'][layer_idx]
        curr_layer['batch'] = s['batch']

        plane_dims[layer_name] = curr_layer
        prev_layer = curr_layer

    total_planes_size = compute_planes_size(plane_dims) #plane_dims is updated in-place

    pprint(s)
    pprint(plane_dims)
    print 'total planes size:', total_planes_size, 'MB'
    
