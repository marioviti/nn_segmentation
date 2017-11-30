from keras import backend as K
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Conv2DTranspose, concatenate, Cropping2D

K.set_image_data_format('channels_last')

def crop_to(bigger_input_size,smaller_input_size):
    _,bh,bw,_ = bigger_input_size
    _,sh,sw,_ = smaller_input_size
    dh,dw = bh-sh, bw-sw
    q2dw,r2dw,q2dh,r2dh = dw//2 , dw%2, dh//2 , dh%2
    return Cropping2D(cropping=((q2dh,q2dh+r2dh), (q2dw, q2dw+r2dw)), data_format='channels_last')

def crop_concatenate(bigger_input, smaller_input):
    bigger_input_size = bigger_input._keras_shape
    smaller_input_size = smaller_input._keras_shape
    cropped_input = crop_to(bigger_input_size,smaller_input_size)(bigger_input)
    return concatenate([cropped_input,smaller_input],axis=3)

def cnv3x3Relu(filters):
    return Conv2D(filters, (3, 3), activation='relu', padding='valid')

def downsample(pool_size=(2, 2)):
    return MaxPool2D(pool_size=pool_size)

def upsample(filters):
    return Conv2DTranspose(filters, (2, 2), strides=(2,2),
                            padding='valid',
                            activation='relu')

def new_down_level(filters, inputs):
    outputs= downsample()(inputs)
    outputs = cnv3x3Relu(filters)(outputs)
    outputs = cnv3x3Relu(filters)(outputs)
    return outputs

def new_up_level(filters, up_input, right_input):
    outputs = upsample(filters)(up_input)
    outputs = crop_concatenate(right_input,outputs)
    outputs = cnv3x3Relu(filters)(outputs)
    outputs = cnv3x3Relu(filters)(outputs)
    return outputs
