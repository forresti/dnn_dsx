from math import floor
from pprint import pprint

def compute_params_size(layer_dims):
    total_params_size = 0
    for k in layer_dims.keys():
        L = layer_dims[k]
        params_size = L['chans_in'] * L['chans_out'] * L['h'] * L['w'] * 4 #4 bytes/float
        params_size = float(params_size) / 1048576.0 #bytes to MB
        layer_dims[k]['params_size'] = params_size
        total_params_size = total_params_size + params_size 

    return total_params_size

if __name__ == "__main__":

    s = dict() #settings

    #FireNet w/ 8 fire layers
    s['num_fire_layers']=8

    s['incr_freq'] = 2 #increase # filters after every 'this many' layers.
    s['base_expand'] = 64 #number of (1x1_2 + 3x3_2) filters
    s['incr_expand'] = 64 

    s['3x3_expand_pct'] = 0.25 #percentage of (1x1_2 + 3x3_2) that are 3x3_2
    s['compress_expand_ratio'] = 1.0 #ratio of 1x1_1 / (1x1_2 + 3x3_2)

    #FireNet w/ 10 fire layers...
    #TODO

    layer_dims=dict()

    #TODO: assert(0 <= s['3x3_expand_pct'] <= 1)
    #TODO: assert(0 <= s['compress_expand_ratio'] <= 1)
    prev_chans_out=3 #rgb
    for layer_idx in xrange(0, s['num_fire_layers']):
        base_name = "fire%d/" %layer_idx
        #num_1x1_1 = s['base_1x1_1'] + floor(layer_idx/s['incr_freq']) * s['incr_add_1x1_1']
        #num_1x1_2 = s['base_1x1_2'] + floor(layer_idx/s['incr_freq']) * s['incr_add_1x1_2']
        #num_3x3_2 = s['base_3x3_2'] + floor(layer_idx/s['incr_freq']) * s['incr_add_3x3_2'] 

        num_expand = s['base_expand'] + floor(layer_idx/s['incr_freq']) * s['incr_expand']
        num_3x3_2 = int(floor(s['3x3_expand_pct'] * num_expand))
        num_1x1_2 = num_expand - num_3x3_2
        num_1x1_1 = s['compress_expand_ratio'] * num_expand 


        layer_dims[base_name+'conv1x1_1'] = {'chans_in':prev_chans_out, 'chans_out':num_1x1_1, 'h':1, 'w':1}
        layer_dims[base_name+'conv1x1_2'] = {'chans_in':num_1x1_1, 'chans_out':num_1x1_2, 'h':1, 'w':1}
        layer_dims[base_name+'conv3x3_2'] = {'chans_in':num_1x1_1, 'chans_out':num_3x3_2, 'h':3, 'w':3}

        print "compress/expand ratio: ", num_1x1_1 / (num_1x1_2 + num_3x3_2)
        print "1x1-expand / 3x3-expand ratio: ", (num_1x1_2 / num_3x3_2)

        prev_chans_out = num_3x3_2+num_1x1_2

    total_params_size = compute_params_size(layer_dims) #layer_dims is updated in-place

    #pprint(s)
    pprint(layer_dims)
    print 'total params size:', total_params_size, 'MB'
    
