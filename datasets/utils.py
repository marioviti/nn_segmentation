import numpy as np
import time
from keras.utils import np_utils
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform

np.random.seed(int((time.time()*1e6)%1e6))

def rotation(x, theta, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def images_random_rotate(images, rg=0):
    """
        rg : rotation range in deg
    """
    if rg==0:
        return images
    theta = np.deg2rad(np.random.uniform(-rg, rg))
    rotated_images = []
    for image in images:
        if len(image.shape)==2:
            image = np.reshape(image,list(image.shape)+[1])
            rotated = rotation(image,theta)
            rotated = np.reshape(rotated,rotated.shape[0:2])
            rotated_images += [rotated]
        else:
            rotated_images += [rotation(image,theta)]
    return rotated_images

def integral(x):
    int_sum = np.sum(x)
    n = 1
    for s in list(np.shape(x)):
        n*=s
    return int_sum/float(n)

def refuse_batch(batch,score_function=integral,threshold=0.2):
    score = score_function(batch)
    return score > threshold

# from categorical for a batch patch of size
# N_batch,H,W,Classes, inverse operation of to_categorical
def from_categorical(batch_patch):
    return np.argmax(batch_patch,axis=3)

# to categorical for a batch patch of size
# N_batch,H,W
def to_categorical(batch_patch,num_classes=2):
    return np_utils.to_categorical(batch_patch,num_classes=num_classes)

# Normalize to -1,1
def normalize(images, means, std):
    return (images - means)/std

# Conver to numpy from PIL image
def to_numpy(image):
    return np.array(image)/255.

# Conver to numpy from PIL image
def images_to_numpy(images):
    numpy_images = []
    for image in images:
        numpy_images += [to_numpy(image)]
    return numpy_images

# Sample patch at offset_h,offset_h of width=height=w
def get_patch(x,h,w,offset_h,offset_w):
    return x[offset_h:offset_h+h,offset_w:offset_w+w]

# Sample patch from an array of images at the same offset and size
def sample_patches(images,h,w,offset_h=None,offset_w=None):
    offset_h = np.random.randint(h) if offset_h is None else offset_h
    offset_w = np.random.randint(w) if offset_w is None else offset_w
    patches = []
    for image in images:
        patches += [get_patch(image,h,w,offset_h,offset_w)]
    return patches

# Sample batch_size patches from a set of images
def batch_patches(images,batch_size,h,w):
    N_batches = len(images)
    batches = sample_patches(images,h,w)
    for k in range(N_batches):
        batches[k] = np.expand_dims(batches[k], axis=0)
    for i in range(1,batch_size):
        patches = sample_patches(images,h,w)
        for k in range(N_batches):
            patches[k] = np.expand_dims(patches[k], axis=0)
            batches[k] = np.concatenate((batches[k],patches[k]), axis=0)
    return batches


# Concatenate multiple patch_batches
def concatenate_batches_patches(batches_patches):
    N_batches = len(batches_patches)
    bs_ps_concat = batches_patches[0]
    for i in range(1,N_batches):
        bs_ps_concat = np.concatenate((bs_ps_concat,batches_patches[i]), axis=0)
    return bs_ps_concat
