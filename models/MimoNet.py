from layers import *
from serialize import *
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

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

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

    layers['outputs'] = [ feature_mask(4,256,64,classes,layers['up_path'][2],'la1') ]
    layers['outputs'] += [ feature_mask(2,128,64,classes,layers['up_path'][3],'la2') ]
    layers['outputs'] += [ Conv2D(classes, (1, 1), activation='softmax', name='la3')(layers['up_path'][4]) ]
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
la1_counter = 0.0
la2_counter = 0.0

def weighted_categorical_crossentropy(target, output):
    output /= tf.reduce_sum(output,
                            len(output.get_shape()) - 1,
                            True)
    return - tf.reduce_sum(target * tf.log(output),
                           len(output.get_shape()) - 1)

def la1_categorical_crossentropy(target,output):
    global la1_counter
    la1_counter += 1.0
    return weighted_categorical_crossentropy(target,output)/la1_counter

def la2_categorical_crossentropy(target,output):
    global la2_counter
    la2_counter += 1.0
    return weighted_categorical_crossentropy(target,output)/la2_counter

def define_mimonet_inputs_shapes(input_shape):
    h,w,c = input_shape
    return [ input_shape, [h//2,w//2,c], [h//4,w//4,c] ]

def compute_mimonet_inputs(x,shapes):
    x1 = resize(x,shapes[1], anti_aliasing=True)
    x2 = resize(x,shapes[2], anti_aliasing=True)
    return x,x1,x2

class MimoNet(GenericModel):
    def __init__( self, input_shape, classes=2,
                  regularized = False,
                  loss={'la1': la1_categorical_crossentropy,
                              'la2': la2_categorical_crossentropy,
                              'la3': weighted_categorical_crossentropy},
                  loss_weights={'la1': 1.0, 'la2': 0.7 , 'la3': 0.7 },
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
        print('inputs_shape',self.inputs_shape)
        print('outputs_shape', self.outputs_shape)

    def fit( self, x_train, y_train, batch_size=1, epochs=1, cropped=False ):
        x_train1,x_trai2,x_trai3 = compute_mimonet_inputs(x_train,self.input_shapes)
        out_shape = self.outputs_shape[0]
        y_train = y_train if cropped else crop_receptive(y_train, out_shape)
        return GenericModel.fit( self,
                                 {   'in1': x_train1,
                                     'in2': x_train2,
                                     'in3': x_train3
                                 },
                                 {   'la1': y_train,
                                     'la2': y_train,
                                     'la3': y_train
                                 },
                                 epochs=epochs,
                                 batch_size=batch_size )

    def evaluate( self, x_test,  y_test,  batch_size=1, cropped=False ):
        x_test1,x_test2,x_test3 = compute_mimonet_inputs(x_test,self.input_shapes)
        out_shape = self.outputs_shape[0]
        y_test = y_test if cropped else crop_receptive(y_test, out_shape)
        return GenericModel.evaluate( self,
                                     {   'in1': x_test1,
                                         'in2': x_test2,
                                         'in3': x_test3
                                     },
                                     {   'la1': y_test,
                                         'la2': y_test,
                                         'la3': y_test
                                     }, 
                                     batch_size=batch_size )
    
if __name__=='__main__':
    mimo = MimoNet([[350,350,3],[175,175,3],[87,87,3]])
    
