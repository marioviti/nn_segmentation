from layers import *

import numpy as np

from keras import backend as K
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import SGD,Adam

import tensorflow as tf

K.set_image_data_format('channels_last')

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def weighted_cross_entropy(w):
    def crossentropy(y_true,y_pred):
        return binary_crossentropy(y_true,y_pred)
    return crossentropy

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

class Unet():
    def __init__(self, input_size):
        """
        params: input_size, channels_last (h,w,c)
        """
        model, layers = get_unet_model(input_size)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.layers = layers
        self.input_shape = layers['inputs']._keras_shape
        self.outputs_shape = layers['outputs']._keras_shape

        self.model = model
        #self.model.compile(loss=weighted_cross_entropy(1), optimizer=sgd, metrics=['accuracy'])
        self.model.compile(optimizer=Adam(lr=1e-5), loss=binary_crossentropy, metrics=['accuracy'])

    def fit(self,x_train,y_train,epochs=5,batch_size=5):
        self.model.fit(x_train, y_train,
                        epochs=epochs,batch_size=batch_size)

    def evaulate(self,x_test, y_test, batch_size=2):
        self.score = self.model.evaluate(x_test, y_test, batch_size=batch_size)
        return self.score

    def get_model_output_shape(self):
        return self.outputs_shape[1:]

    def get_model_input_shape(self):
        return self.input_shape

    def predict_batch(self,x,batch_size=1,verbose=0):
        return self.model.predict(x, batch_size=batch_size, verbose=verbose)

    def predict(self,x,verbose=0):
        return self.model.predict(x.reshape([1]+list(x.shape)), batch_size=1, verbose=verbose)

#def main():
#    # Generate dummy data
#    input_size = [350,350,3]
#    unet = Unet(input_size)
#    _,h,w,c = unet.outputs_shape
#    x_train = np.random.random([5] + input_size)
#    y_train = np.random.randint(2, size=([5,h,w,c]))
#    x_test = np.random.random([2] + input_size)
#    y_test = np.random.randint(2, size=([2,h,w,c]))
#
#    unet.fit(x_train,y_train)
#    unet.evaulate(x_train,y_train)
#    y_hat = unet.predict(x_train[0])
#    print(y_hat.shape)
#    print(unet.get_model_output_shape())
#    print(unet.score)
#
#if __name__ == '__main__':
#    main()
