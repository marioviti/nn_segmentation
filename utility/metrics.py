import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion

def dice(y_true, y_pred):
    """
    Attention: 
    y_true can be weighted to modify learning therefore 
    apply sign to get back to labels
    y_pred have to be rounded to nearest integer to obtain labels.
    """
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f =  y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def true_pos(y_true, y_pred):
    return np.sum(y_true * y_pred)

def false_pos(y_true, y_pred):
     return np.sum(y_true * (1. - (y_pred)))

def false_neg(y_true, y_pred):
     return np.sum((1. - y_true) * (y_pred))

def precision(y_true, y_pred):
     return true_pos(y_true, y_pred) / \
         (true_pos(y_true, y_pred) + false_pos(y_true, y_pred))

def Pc(Yor,Y_hator,tetha=5,c=1.0):
    Y = np.copy(Yor)
    Y_hat = np.copy(Y_hator)
    Y[Y!=c] = 0
    Y[Y==c] = 1.0
    Y_hat[Y_hat!=c] = 0
    Y_hat[Y_hat==c] = 1.0
    Bgt = Y - binary_erosion(Y,structure=np.ones((3,3)))
    Bps = Y_hat - binary_erosion(Y_hat,structure=np.ones((3,3)))
    D = distance_transform_edt(1-Bgt)
    D_Bpd = D[Bps==1.0]
    return np.sum(D_Bpd<tetha)/(float(np.sum(Bps==1.0)) + 1)