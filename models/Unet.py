from layers import *
from serialize import *

import numpy as np

from keras import backend as K
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.layers import ZeroPadding2D

import tensorflow as tf

K.set_image_data_format('channels_last')

def crop_receptive(batch_y, model_output_size):
    """
        Get a cropped batch to fit the perceptive field,
        the resulting output shape is n,hy,wy,cy.

        args:
            - batch_y (numpy array) y.shape : n,hx,wx,cy
            - model_output_size (list) : hy,wy,cy
    """
    n,hx,wx,cy = batch_y.shape
    hy,wy,cy = model_output_size
    dhq, dhr = (hx-hy)//2, (hx-hy)%2
    dwq, dwr = (wx-wy)//2, (wx-wy)%2
    return batch_y[:, dhq: hx - (dhq + dhr), dwq: wx - (dwq + dwr) ]

def expand_receptive(batch_y, model_input_shape):
    """
        Get a expantded batch to fit the model_input_shape hx and wx,
        the resulting output shape is n,hx,wx,cy.

        args:
            - batch_y (numpy array) y.shape : n,hy,wy,cy
            - model_input_shape (list) : hx,wx,cx
    """
    hx,wx,cx = model_input_shape
    n,hy,wy,cy = batch_y.shape
    dhq, dhr = (hx-hy)//2, (hx-hy)%2
    dwq, dwr = (wx-wy)//2, (wx-wy)%2
    y_expanded = np.zeros((n,hx,wx,cy))
    y_expanded[:, dhq: - (dhq + dhr), dwq: - (dwq + dwr) ] = batch_y
    return y_expanded

# TODO look at this if is working for categoriacal labels
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

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

    layers['outputs'] = Conv2D(2, (1, 1), activation='softmax')(layers['up_path'][4])
    return layers

def get_unet_model(input_size):
    layers = define_unet_layers(input_size)
    model = Model(inputs=[layers['inputs']], outputs=[layers['outputs']])
    return model, layers

class Unet():
    def __init__( self, input_shape, loss=categorical_crossentropy, \
                  metrics=[dice_coef], optimizer=Adam(lr=1e-5) ):
        """
        params:
            inputs_shape: (tuple) channels_last (h,w,c) of input image.
            metrics:    (tuple) metrics function for evaluation.
            optimizer:  (function) Optimization strategy.
        """
        model, layers = get_unet_model(input_shape)
        self.model = model
        self.layers = layers

        self.inputs_shape = layers['inputs']._keras_shape
        self.outputs_shape = layers['outputs']._keras_shape

        self.loss= loss
        self.metrics = metrics
        self.optimizer = optimizer

        self.compile_model()

    @property
    def optimizer(self):
        return self.optimizer

    @optimizer.setter
    def optimizer(self, v):
        self.optimizer = v

    @property
    def metrics(self):
        return self.metrics

    @metrics.setter
    def metrics(self, v):
        self.metrics = v

    @property
    def input_shape(self):
        return self.inputs_shape[1:]

    @property
    def output_shape(self):
        return self.outputs_shape[1:]

    def save_model(self, name=None):
        self.name = self.name if name is None else name
        save_to( self.model,self.name )

    def load_model(self, name=None):
        self.name = self.name if name is None else name
        self.model = load_from( self.name )
        self.compile_model()

    def compile_model(self):
        self.model.compile( optimizer=self.optimizer, \
                            loss=self.loss, metrics=self.metrics )

    def fit( self, x_train, y_train, batch_size=1, epochs=1, cropped=False ):
        out_shape = self.output_shape
        y_train = y_train if cropped else crop_receptive(y_train, out_shape)
        print (out_shape)
        print (y_train.shape)
        self.model.fit( x_train, y_train, \
                        epochs=epochs, batch_size=batch_size )

    def evaulate( self, x_test,  y_test,  batch_size=1, cropped=False ):
        out_shape = self.output_shape
        y_test = y_test if cropped else crop_receptive(y_test, out_shape)
        self.score = self.model.evaluate(x_test, y_test, batch_size=batch_size )
        return self.score

    def predict( self, x, batch_size=1, verbose=0 ):
        return self.model.predict( x, batch_size=batch_size, verbose=verbose )

    def predict_mask( self, x, verbose=0 ):
        hx,wx,cx = self.input_shape
        hy,wy,cy = self.output_shape
        patch = self.predict(x.reshape([1,hx,wx,cx]), verbose=verbose)
        return expand_receptive(patch,[hx,wx,cy])

    def predict_and_stich( self, X, stride=1 ):
        hX,wX,cX = X.shape
        hx,wx,cx = self.input_shape
        hy,wy,cy = self.output_shape
        Y = np.zeros((hX,wX,cy))
        for i in range(hX-hx,stride):
            for j in range(wX-wx,stride):
                patch_x = X[ i:i+hx, j:j+wx ]
                Y[ i:i+hx, j:j+wx ] += predict_mask(patch_x)
        return Y


def main():
    # Generate dummy data
    input_size = [350,350,3]
    unet = Unet(input_size)
    _,h,w,c = unet.outputs_shape
    x_train = np.random.random([5,350,350,3])
    y_train = np.random.randint(2, size=([5,350,350,2]))

    unet.fit(x_train,y_train)
    unet.evaulate(x_train,y_train)
    y_hat = unet.predict(x_train[0])
    print(y_hat.shape)
    print(unet.get_model_output_shape())
    print(unet.score)

if __name__ == '__main__':
    main()

#binary_crossentropy
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#self.model.compile(loss=weighted_cross_entropy(1), optimizer=sgd, metrics=['accuracy'])
#self.model.compile(optimizer=Adam(lr=1e-5), loss=binary_crossentropy, metrics=['accuracy'])
