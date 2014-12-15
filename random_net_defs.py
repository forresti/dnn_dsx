
def dataLayerStr_deploy(batch, channels, height, width):
    dataLayerStr = '\n \
    name: "AlexNet" \n \
    input: "data_layer" \n \
    input_dim: ' + str(batch) + '\n \
    input_dim: ' + str(channels) + '\n \
    input_dim: ' + str(height) + '\n \
    input_dim: ' + str(width) 
    return dataLayerStr

def convLayerStr(name, bottom, top, num_output, kernel_size, stride):

    #AlexNet uses bias_filler=0 for some layers and 1 for other layers
    #NiN uses bias_filler=0 for all layers.

    convStr = ' \n \
    layers { \n \
      name: "' + name + '" \n \
      type: CONVOLUTION \n \
      bottom: "' + bottom  + '" \n \
      top: "' + top + '" \n \
      blobs_lr: 1 \n \
      blobs_lr: 2 \n \
      weight_decay: 1 \n \
      weight_decay: 0 \n \
      convolution_param { \n \
        num_output: ' + str(num_output) + ' \n \
        kernel_size: ' + str(kernel_size) + ' \n \
        stride: ' + str(stride) + '\n \
        weight_filler { \n \
          type: "gaussian" \n \
          std: 0.01 \n \
        } \n \
        bias_filler { \n \
          type: "constant" \n \
          value: 0 \n \
        } \n \
      } \n \
    }' 
    return convStr

def poolLayerStr(name, bottom, top, kernel_size, stride, poolType):
    poolStr = '\n \
    layers {\n \
      name: "' + name + '"\n \
      type: POOLING\n \
      bottom: "' + bottom + '"\n \
      top: "' + top + '"\n \
      pooling_param {\n \
        pool: ' + poolType + '\n \
        kernel_size: ' + str(kernel_size) + '\n \
        stride: ' + str(stride) + '\n \
      }\n \
    }'
    return poolStr


def reluLayerStr(name, bottom, top):
    reluStr = '\n \
    layers {\n \
      name: "' + name + '"\n \
      type: RELU\n \
      bottom: "' + bottom +'"\n \
      top: "' + top + '"\n \
    }'
    return reluStr


def lrnLayerStr(name, bottom, top, local_size):
    #all AlexNet LRN layers use these alpha and beta params.
    #all AlexNet and GoogLeNet LRN layers use local_size=5
    lrnStr = '\n \
    layers { \n \
      name: "' + name + '"\n \
      type: LRN \n \
      bottom: "' + bottom  + '"\n \
      top: "' + top + '"\n \
      lrn_param {\n \
        local_size: ' + str(local_size) + '\n \
        alpha: 0.0001\n \
        beta: 0.75\n \
      }\n \
    }'
    return lrnStr

