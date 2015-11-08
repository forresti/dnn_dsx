



if __name__ == "__main__":

    base_1x1_1 = 128
    base_1x1_2 = 128 #TODO: vary... these are cheap.
    base_3x3_2 = 64

    incr_add_1x1_1 = 128
    incr_add_1x1_2 = 128 #TODO: vary... these are cheap.
    incr_add_3x3_2 = 64

    for layer_idx in xrange(0, 5):
        base_name = "Fire%d/" %layer_idx
        num_1x1_1 = base_1x1_1 + layer_idx*incr_add_1x1_1
        num_1x1_2 = base_1x1_2 + layer_idx*incr_add_1x1_2
        num_3x3_2 = base_3x3_2 + layer_idx*incr_add_3x3_2 

        st = base_name + ' '
        st += '1x1_1: ' + str(num_1x1_1)
        st += ', 1x1_2: ' + str(num_1x1_2)
        st += ', 3x3_2: ' + str(num_3x3_2) 
        print st

