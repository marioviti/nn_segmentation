from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Conv2DTranspose, concatenate, Cropping2D
from keras import backend as K
import tensorflow as tf

K.set_image_data_format('channels_last')
# When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in  data_format="channels_last"

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

def define_unet_layers(input_shape):
    """
    Use the functional API to define the model
    https://keras.io/getting-started/functional-api-guide/
    params: input_shape (h,w,channels)
    """
    layers = {  'inputs' : None,
                'down_path' : {},
                'bottle_neck' : None,
                'up_path' : {},
                'outputs' : None }

    layers['inputs'] = Input(input_shape)

    layers['down_path'][4] = cnv3x3Relu(64)(layers['inputs'])
    layers['down_path'][4] = cnv3x3Relu(64)(layers['down_path'][4])
    layers['down_path'][3] = new_down_level(128,layers['down_path'][4])
    layers['down_path'][2] = new_down_level(256,layers['down_path'][3])
    layers['down_path'][1] = new_down_level(512,layers['down_path'][2])

    layers['bottle_neck'] = new_down_level(1024,layers['down_path'][1])

    layers['up_path'][1] = new_up_level(512,layers['bottle_neck'],layers['down_path'][1])
    layers['up_path'][2] = new_up_level(256,layers['up_path'][1],layers['down_path'][2])
    layers['up_path'][3] = new_up_level(128,layers['up_path'][2],layers['down_path'][3])
    layers['up_path'][4] = new_up_level(64,layers['up_path'][3],layers['down_path'][4])

    layers['outputs'] = Conv2D(1, (1, 1), activation='sigmoid')(layers['up_path'][4])
    return layers


def get_unet_model(input_size):
    layers = define_unet_layers(input_size)
    model = Model(inputs=[layers['inputs']], outputs=[layers['outputs']])
    return model, layers

def test():
    model, layers = get_unet_model((572,572,3))

def main():
    test()

if __name__ == '__main__':
    main()
