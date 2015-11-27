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

    base_1x1_1 = 128
    base_1x1_2 = 128 #TODO: vary... these are cheap.
    base_3x3_2 = 64

    incr_add_1x1_1 = 128
    incr_add_1x1_2 = 128 #TODO: vary... these are cheap.
    incr_add_3x3_2 = 64

    layer_dims=dict()

    prev_chans_out=3 #rgb
    for layer_idx in xrange(0, 4):
        base_name = "fire%d/" %layer_idx
        num_1x1_1 = base_1x1_1 + layer_idx*incr_add_1x1_1
        num_1x1_2 = base_1x1_2 + layer_idx*incr_add_1x1_2
        num_3x3_2 = base_3x3_2 + layer_idx*incr_add_3x3_2 

        layer_dims[base_name+'conv1x1_1'] = {'chans_in':prev_chans_out, 'chans_out':num_1x1_1, 'h':1, 'w':1}
        layer_dims[base_name+'conv1x1_2'] = {'chans_in':num_1x1_1, 'chans_out':num_1x1_2, 'h':1, 'w':1}
        layer_dims[base_name+'conv3x3_2'] = {'chans_in':num_1x1_1, 'chans_out':num_3x3_2, 'h':3, 'w':3}

        prev_chans_out = num_3x3_2+num_1x1_2

        st = base_name + ' '
        st += '1x1_1: ' + str(num_1x1_1)
        st += ', 1x1_2: ' + str(num_1x1_2)
        st += ', 3x3_2: ' + str(num_3x3_2) 
        #print st


    total_params_size = compute_params_size(layer_dims) #layer_dims is updated in-place
    pprint(layer_dims)
    print 'total params size:', total_params_size
    
