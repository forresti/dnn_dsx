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

    '''
    #FireNet_default (not including beginning and ending conv layers)
    s['num_fire_layers']=4

    s['base_1x1_1'] = 128
    s['base_1x1_2'] = 128 #TODO: vary... these are cheap.
    s['base_3x3_2'] = 64

    s['incr_freq'] = 1 #increase # filters after every 'this many' layers.
    s['incr_add_1x1_1'] = 128
    s['incr_add_1x1_2'] = 128 #TODO: vary... these are cheap.
    s['incr_add_3x3_2'] = 64
    '''

    #FireNet w/ 8 fire layers
    s['num_fire_layers']=8

    s['base_1x1_1'] = 64
    s['base_1x1_2'] = 256
    s['base_3x3_2'] = 64

    s['incr_freq'] = 2 #increase # filters after every 'this many' layers.
    s['incr_add_1x1_1'] = 64
    s['incr_add_1x1_2'] = 256 
    s['incr_add_3x3_2'] = 64

    '''    
    #FireNet w/ 10 fire layers
    s['num_fire_layers']=10

    s['base_1x1_1'] = 64
    s['base_1x1_2'] = 64 
    s['base_3x3_2'] = 32

    s['incr_freq'] = 2 #increase # filters after every 'this many' layers.
    s['incr_add_1x1_1'] = 64
    s['incr_add_1x1_2'] = 64 
    s['incr_add_3x3_2'] = 16
    '''

    layer_dims=dict()

    prev_chans_out=3 #rgb
    for layer_idx in xrange(0, s['num_fire_layers']):
        base_name = "fire%d/" %layer_idx
        num_1x1_1 = s['base_1x1_1'] + floor(layer_idx/s['incr_freq']) * s['incr_add_1x1_1']
        num_1x1_2 = s['base_1x1_2'] + floor(layer_idx/s['incr_freq']) * s['incr_add_1x1_2']
        num_3x3_2 = s['base_3x3_2'] + floor(layer_idx/s['incr_freq']) * s['incr_add_3x3_2'] 

        layer_dims[base_name+'conv1x1_1'] = {'chans_in':prev_chans_out, 'chans_out':num_1x1_1, 'h':1, 'w':1}
        layer_dims[base_name+'conv1x1_2'] = {'chans_in':num_1x1_1, 'chans_out':num_1x1_2, 'h':1, 'w':1}
        layer_dims[base_name+'conv3x3_2'] = {'chans_in':num_1x1_1, 'chans_out':num_3x3_2, 'h':3, 'w':3}

        prev_chans_out = num_3x3_2+num_1x1_2

    total_params_size = compute_params_size(layer_dims) #layer_dims is updated in-place

    pprint(s)
    pprint(layer_dims)
    print 'total params size:', total_params_size, 'MB'
    
