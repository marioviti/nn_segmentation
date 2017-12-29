import numpy as np
import time

np.random.seed(int((time.time()*1e6)%1e6))

def get_patch(x,offset_h,h,offset_w,w):
    """
        get a patch or sub-image or ROI or sub matrix of size h x w
        starting from offset_h and offset_w.

        args:
            - x (numpy array): image or matrix (x.shape = h,w,c channels last)
    """
    return x[offset_h:offset_h+h,offset_w:offset_w+w]

def sample_patches(images,h,w,offset_h=None,offset_w=None):
    offset_h = np.random.randint(h) if offset_h is None else offset_h
    offset_w = np.random.randint(w) if offset_w is None else offset_w
    patches = []
    for image in images:
        patches += [get_patch(image,offset_h,h,offset_w,w)]
    return patches
