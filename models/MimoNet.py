from layers import *
from serialize import *
from metrics_and_losses import *
from GenericModel import GenericModel

import numpy as np

from keras import backend as K
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.utils import to_categorical
from keras.optimizers import SGD,Adam
from keras.layers import ZeroPadding2D

from skimage.transform import resize
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

def define_mimonet_layers(input_shape, classes, regularized=False):
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

    layers['inputs'] = [Input(input_shape[0],name='in1'),Input(input_shape[1],name='in2'),Input(input_shape[2],name='in3')]
    layers['down_path'][4] = cnv3x3Relu(64,regularized=regularized)(layers['inputs'][0])
    layers['down_path'][4] = cnv3x3Relu(64,regularized=regularized)(layers['down_path'][4])
    
    layers['down_path'][3] = crop_concatenate(layers['inputs'][1], 
                                              new_down_level(128,layers['down_path'][4],regularized=regularized))
    
    layers['down_path'][2] =  crop_concatenate(layers['inputs'][2],
                                               new_down_level(256,layers['down_path'][3],regularized=regularized))
    
    layers['down_path'][1] = new_down_level(512,layers['down_path'][2],regularized=regularized)

    layers['bottle_neck'] = new_down_level(1024,layers['down_path'][1],regularized=regularized)

    layers['up_path'][1] = new_up_level(512,layers['bottle_neck'],layers['down_path'][1],regularized=regularized)
    layers['up_path'][2] = new_up_level(256,layers['up_path'][1],layers['down_path'][2],padding='same',regularized=regularized)
    layers['up_path'][3] = new_up_level(128,layers['up_path'][2],layers['down_path'][3],padding='same',regularized=regularized)
    layers['up_path'][4] = new_up_level(64,layers['up_path'][3],layers['down_path'][4],regularized=regularized)

    auxla1, la1 = feature_mask(4,256,64,classes,layers['up_path'][2],'la1')
    auxla2, la2 = feature_mask(2,128,64,classes,layers['up_path'][3],'la2')
    auxla3 = layers['up_path'][4]
    layers['outputs'] = [ la1,la2 ]
    layers['outputs'] += [ Conv2D(classes, (1, 1), activation='softmax', name='la3')(auxla3) ]
    
    l0 = crop_concatenate(auxla1, auxla2)
    l0 = crop_concatenate(l0,auxla3)
    l0 = cnv3x3Relu(64,regularized=regularized, padding='same')(l0)
    l0 = cnv3x3Relu(32,regularized=regularized, padding='same')(l0)
    layers['outputs'] += [ Conv2D(classes, (1, 1), activation='softmax', name='l0')(l0) ]
    return layers

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

global la1_counter
global la2_counter
global la3_counter
la1_counter = 1.0
la2_counter = 1.0
la3_counter = 1.0

def l0_categorical_crossentropy(target,output):
    return softmax_categorical_crossentropy(target,output)

def la1_categorical_crossentropy(target,output):
    global la1_counter
    la1_counter += 0.75
    return softmax_categorical_crossentropy(target,output)/la1_counter

def la2_categorical_crossentropy(target,output):
    global la2_counter
    la2_counter += 0.5
    return softmax_categorical_crossentropy(target,output)/la2_counter

def la3_categorical_crossentropy(target,output):
    global la3_counter
    la3_counter += 0.125
    return softmax_categorical_crossentropy(target,output)/la3_counter

def define_mimonet_inputs_shapes(input_shape):
    h,w,c = input_shape
    return [ input_shape, [h//2,w//2,c], [h//4,w//4,c] ]

def compute_mimonet_inputs(x,shapes):
    """
    MimoNet uses multiple inputs at different scales: this function
    helps to resize data to fit in each input
    """
    n = x.shape[0]
    h1,w1,c = shapes[1]
    h2,w2,c = shapes[2]
    x1 = np.zeros((n,h1,w1,c), dtype=x.dtype)
    x2 = np.zeros((n,h2,w2,c), dtype=x.dtype)
    for i in range(n):
        x1[i,:,:,:] = resize(x[i,:,:,:],[h1,w1,c])
        x2[i,:,:,:] = resize(x[i,:,:,:],[h2,w2,c])
    return x,x1,x2

class MimoNet(GenericModel):
    def __init__( self, input_shape, classes=2,
                  regularized = False,
                  loss={'la1': la1_categorical_crossentropy,
                        'la2': la2_categorical_crossentropy,
                        'la3': la3_categorical_crossentropy,
                        'l0' : l0_categorical_crossentropy
                  },
                  loss_weights={'la1': 1.0, 'la2': 1.0 , 'la3': 1.0 },
                  metrics=[dice_coef],
                  optimizer=Adam(lr=1e-5) ):
        """
        params:
            inputs_shape: (tuple) channels_last (h,w,c) of input image.
            metrics:    (tuple) metrics function for evaluation.
            optimizer:  (function) Optimization strategy.
        """
        input_shapes = define_mimonet_inputs_shapes(input_shape)
        layers = define_mimonet_layers(input_shapes, classes, regularized=regularized)
        self.layers = layers
        self.classes = classes
        inputs, outputs = layers['inputs'],layers['outputs']
        GenericModel.__init__(self, inputs, outputs, loss, metrics, optimizer,
                                loss_weights=loss_weights)

    def fit( self, x_train, y_train, batch_size=1, epochs=1, cropped=False ):
        x_train1,x_train2,x_train3 = compute_mimonet_inputs(x_train,self.inputs_shape)
        out_shape = self.outputs_shape[0]
        y_train = y_train if cropped else crop_receptive(y_train, out_shape)
        return GenericModel.fit( self,
                                 {   'in1': x_train1,
                                     'in2': x_train2,
                                     'in3': x_train3
                                 },
                                 {   'la1': y_train,
                                     'la2': y_train,
                                     'la3': y_train,
                                      'l0': y_train
                                 },
                                 epochs=epochs,
                                 batch_size=batch_size )

    def evaluate( self, x_test,  y_test,  batch_size=1, cropped=False ):
        x_test1,x_test2,x_test3 = compute_mimonet_inputs(x_test,self.inputs_shape)
        out_shape = self.outputs_shape[0]
        y_test = y_test if cropped else crop_receptive(y_test, out_shape)
        return GenericModel.evaluate( self,
                                         {   'in1': x_test1,
                                             'in2': x_test2,
                                             'in3': x_test3
                                         },
                                         {   'la1': y_test,
                                             'la2': y_test,
                                             'la3': y_test,
                                              'l0': y_test
                                         }, 
                                         batch_size=batch_size )
    
    def predict(self, x, batch_size=1, verbose=0):
        x1,x2,x3 = compute_mimonet_inputs(x,self.inputs_shape)
        out_shape = self.outputs_shape[0]
        ys = GenericModel.predict(self,
                                    {   'in1': x1,
                                        'in2': x2,
                                        'in3': x3
                                    }, 
                                    batch_size=batch_size, 
                                    verbose=verbose )
        return ys[3]
    
