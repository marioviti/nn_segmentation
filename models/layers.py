from keras import backend as K
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Conv2DTranspose, concatenate, Cropping2D, BatchNormalization
from keras import regularizers

K.set_image_data_format('channels_last')

def batch_normalization():
    """
    For instance, after a Conv2D layer with data_format="channels_last",
    set axis=-1 in BatchNormalization.
    """
    return BatchNormalization(axis=-1)

def crop_to(bigger_input_size,smaller_input_size):
    _,bh,bw,_ = bigger_input_size
    _,sh,sw,_ = smaller_input_size
    dh,dw = bh-sh, bw-sw
    q2dw,r2dw,q2dh,r2dh = dw//2 , dw%2, dh//2 , dh%2
    return Cropping2D( cropping=((q2dh,q2dh+r2dh), (q2dw, q2dw+r2dw)),
                       data_format='channels_last')

def crop_concatenate(bigger_input, smaller_input):
    """
    Implement copy crop layer of Uner (skip layer)
    """
    bigger_input_size = bigger_input._keras_shape
    smaller_input_size = smaller_input._keras_shape
    cropped_input = crop_to(bigger_input_size,smaller_input_size)(bigger_input)
    return concatenate([cropped_input,smaller_input],axis=3)

def cnv3x3Relu(filters, regularized=False, padding='valid'):
    layer = Conv2D( filters, (3,3),
                    kernel_regularizer=regularizers.l2(0.01),
                    activation='relu',
                    padding=padding) \
            if regularized else \
            Conv2D( filters, (3, 3),
                    activation='relu',
                    padding=padding)
    return layer

def downsample(pool_size=(2, 2)):
    return MaxPool2D(pool_size=pool_size)

def upsample(filters, padding='valid', kernel=(2,2), strides=(2,2)):
    return Conv2DTranspose(filters, kernel,
                           strides=strides,
                           padding=padding,
                           activation='relu')

def feature_mask(up_factor, up_filters, filters, ouptut_classes, inputs, name,
                 regularized=False, padding='valid'):
    """
        Feature mask layer block
    """
    aux = upsample( up_filters,
                        kernel = (up_factor,up_factor),
                        strides = (up_factor,up_factor),
                        padding = padding)(inputs)
    aux = cnv3x3Relu(filters,
                         padding=padding,
                         regularized=regularized)(aux)
    outputs = cnv3x3Relu(filters,
                         padding=padding,
                         regularized=regularized)(aux)
    outputs = Conv2D(ouptut_classes, (1, 1), activation='softmax', name=name)(outputs)
    print('aux confirmed')
    return aux, outputs

def new_down_level(filters, inputs, regularized=False, padding='valid'):
    """
        Contracting layer block
    """
    outputs= downsample()(inputs)
    outputs = cnv3x3Relu(filters,
                         padding=padding,
                         regularized=regularized)(outputs)
    outputs = cnv3x3Relu(filters,
                         padding=padding,
                         regularized=regularized)(outputs)
    return outputs

def new_up_level(filters, up_input, right_input,
                 regularized=False, padding='valid'):
    """
        Decontracting layer block
    """
    outputs = upsample(filters,
                       padding=padding)(up_input)
    outputs = crop_concatenate(right_input,outputs)
    outputs = cnv3x3Relu(filters,
                         padding=padding,
                         regularized=regularized)(outputs)
    outputs = cnv3x3Relu(filters,
                         padding=padding,
                         regularized=regularized)(outputs)
    return outputs
