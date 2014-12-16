
'''
TODO:
- dropout
- accuracy + softmax

'''

def dataLayerStr_deploy(batch, channels, height, width):
    dataStr = '\n \
    name: "RandomNet" \n \
    input: "data_layer" \n \
    input_dim: ' + str(batch) + '\n \
    input_dim: ' + str(channels) + '\n \
    input_dim: ' + str(height) + '\n \
    input_dim: ' + str(width) 
    return dataStr

def dataLayerStr_trainval(train_dbType, train_db, test_dbType, test_db, batch_size_train, \
                          batch_size_test, crop_size, imagenet_mean_path):
    dataStr = '\n \
    name: "RandomNet" \n \
    layers { \n \
      name: "data_layer" \n \
      type: DATA \n \
      top: "data_layer" \n \
      top: "label" \n \
      data_param { \n \
        source: "' + train_db + '" \n \
        backend: ' + train_dbType + '\n \
        batch_size: ' + str(batch_size_train) + '\n \
      } \n \
      transform_param { \n \
        crop_size: ' + str(crop_size) + '\n \
        mean_file: "' + imagenet_mean_path + '"\n \
        mirror: true \n \
      } \n \
      include: { phase: TRAIN } \n \
    } \n \
    layers { \n \
      name: "data_layer" \n \
      type: DATA \n \
      top: "data_layer" \n \
      top: "label"  \n \
      data_param { \n \
        source: "' + test_db + '" \n \
        backend: ' + test_dbType + ' \n \
        batch_size: ' + str(batch_size_test) + ' \n \
      } \n \
      transform_param { \n \
        crop_size: ' + str(crop_size) + '\n \
        mean_file: "' + imagenet_mean_path + '"\n \
        mirror: false \n \
      } \n \
      include: { phase: TEST } \n \
    }'
    return dataStr

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

def fcLayerStr(name, bottom, top, num_output):
    fcStr = '\n \
    layers{ \n \
      name: "' + name + '"\n \
      type: INNER_PRODUCT \n \
      bottom: "' + bottom  + '"\n \
      top: "' + top + '"\n \
      blobs_lr: 1 \n \
      blobs_lr: 2 \n \
      weight_decay: 1 \n \
      weight_decay: 0 \n \
      inner_product_param { \n \
        num_output: ' + str(num_output) + '\n \
        weight_filler { \n \
          type: "gaussian" \n \
          std: 0.005 \n \
        }\n \
        bias_filler {\n \
          type: "constant"\n \
          value: 0.1\n \
        } \n \
      }\n \
    }'
    return fcStr
