from layers import *
from serialize import *
from GenericModel import GenericModel

import numpy as np

from keras import backend as K
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.utils import to_categorical
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

def define_unet_layers(input_shape, classes, regularized=False):
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

    layers['down_path'][4] = cnv3x3Relu(64,
                            regularized=regularized)(layers['inputs'])
    layers['down_path'][4] = cnv3x3Relu(64,
                            regularized=regularized)(layers['down_path'][4])
    layers['down_path'][3] = new_down_level(128,layers['down_path'][4],regularized=regularized)
    layers['down_path'][2] = new_down_level(256,layers['down_path'][3],regularized=regularized)
    layers['down_path'][1] = new_down_level(512,layers['down_path'][2],regularized=regularized)

    layers['bottle_neck'] = new_down_level(1024,layers['down_path'][1],regularized=regularized)

    layers['up_path'][1] = new_up_level(512,layers['bottle_neck'],layers['down_path'][1],regularized=regularized)
    layers['up_path'][2] = new_up_level(256,layers['up_path'][1],layers['down_path'][2],regularized=regularized)
    layers['up_path'][3] = new_up_level(128,layers['up_path'][2],layers['down_path'][3],regularized=regularized)
    layers['up_path'][4] = new_up_level(64,layers['up_path'][3],layers['down_path'][4],regularized=regularized)

    layers['outputs'] = Conv2D(classes, (1, 1), activation='softmax')(layers['up_path'][4])
    return layers

def get_unet_model(input_size,classes,regularized=False):
    layers = define_unet_layers(input_size,classes,regularized=regularized)
    model = Model(inputs=[layers['inputs']], outputs=[layers['outputs']])
    return model, layers

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

def weighted_categorical_crossentropy(target, output):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # scale preds so that the class probas of each sample sum to 1
    output /= tf.reduce_sum(output,
                            len(output.get_shape()) - 1,
                            True)
    # manual computation of crossentropy
    #_epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
    #output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return - tf.reduce_sum(target * tf.log(output),
                           len(output.get_shape()) - 1)


class Unet(GenericModel):
    def __init__( self, input_shape, classes=2,
                  regularized = False,
                  loss=weighted_categorical_crossentropy,
                  metrics=[dice_coef], optimizer=Adam(lr=1e-5) ):
        """
        params:
            inputs_shape: (tuple) channels_last (h,w,c) of input image.
            metrics:    (tuple) metrics function for evaluation.
            optimizer:  (function) Optimization strategy.
        """
        layers = define_unet_layers(input_shape, classes, regularized=regularized)
        self.layers = layers
        self.classes = classes
        inputs, outputs = [layers['inputs']],[layers['outputs']]
        GenericModel.__init__(self, inputs, outputs, loss, metrics, optimizer)

    def fit( self, x_train, y_train, batch_size=1, epochs=1, cropped=False ):
        out_shape = self.outputs_shape[0]
        y_train = y_train if cropped else crop_receptive(y_train, out_shape)
        return GenericModel.fit( self, x_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size )

    def evaluate( self, x_test,  y_test,  batch_size=1, cropped=False ):
        out_shape = self.outputs_shape[0]
        y_test = y_test if cropped else crop_receptive(y_test, out_shape)
        return GenericModel.evaluate(self,x_test, y_test, batch_size=batch_size )
