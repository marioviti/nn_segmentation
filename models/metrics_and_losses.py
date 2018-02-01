from keras import backend as K
from skimage.transform import resize
import tensorflow as tf

K.set_image_data_format('channels_last')

def true_pos(y_true, y_pred):
    return K.sum(y_true * K.round(y_pred))

def false_pos(y_true, y_pred):
     return K.sum(y_true * (1. - K.round(y_pred)))

def false_neg(y_true, y_pred):
     return K.sum((1. - y_true) * K.round(y_pred))

def precision(y_true, y_pred):
     return true_pos(y_true, y_pred) / \
         (true_pos(y_true, y_pred) + false_pos(y_true, y_pred))
        
def PSNR(y_true, y_pred):
    shape = y_pred.get_shape()
    return K.sum((y_true - K.round(y_pred)))

def dice_coef(y_true, y_pred):
    """
    Attention: 
    y_true can be weighted to modify learning therefore 
    apply sign to get back to labels
    y_pred have to be rounded to nearest integer to obtain labels.
    """
    smooth = 1.
    y_true_f = K.flatten(K.sign(y_true))
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def softmax_categorical_crossentropy(target, output):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # manual computation of crossentropy
    return - tf.reduce_sum(target * tf.log(output),len(output.get_shape())-1)