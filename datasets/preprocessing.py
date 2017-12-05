import numpy as np
from keras.preprocessing.image import random_rotation, apply_transform, transform_matrix_offset_center, random_zoom, load_img
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.measurements import label

import PIL
from matplotlib import pyplot as plt

def morphological_weights(binary_image, sigma=15):
    """
        This function implements the weights used to enforce separation
        of touching object of the same class.
        args:
            - binary_image (numpy.array) : binary_image with segmentation map
            - sigma (float) : maximal width of boundaries in pixels
    """
    regions, number_of_regions = label(binary_image)
    w,h = binary_image.shape
    morphological_weights_map = np.ones((w,h))
    number_of_regions = np.max(regions)
    i_th_map = np.ones((w,h))
    i_th_map[regions==1] = 0
    stack = i_th_map
    # calculate distance function for each cell in image
    for i in range(1,number_of_regions+1):
        i_th_map = np.ones((w,h))
        i_th_map[regions==i] = 0
        distance_map = distance_transform_edt(i_th_map)
        stack = np.dstack((distance_map,stack))

    # for each pixel sort distances and get the top 2-3.
    for x in range(w):
        for y in range(h):
            nearity = np.sort(stack[x,y,:])
            if number_of_regions>2:
                nearest, second_nearest, third_nearst = nearity[0:3]
                distance = ((nearest + second_nearest + third_nearst)**2)/float(sigma)**2
                morphological_weights_map[x,y] = np.exp(-distance)
            else:
                nearest, second_nearest = nearity[0:2]
                distance = ((nearest + second_nearest)**2)/float(sigma)**2
                morphological_weights_map[x,y] = np.exp(-distance)
    return morphological_weights_map


def zoom(x, zoom_ratio, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='bilinear', cval=0.):
    """Performs a spatial zoom of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        zoom_ratio: zoom weight.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Zoomed Numpy image tensor.
    """
    zoom_matrix = np.array([[zoom_ratio, 0, 0],
                            [0, zoom_ratio, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def rotation(x, dg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='bilinear', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Adguments
        x: Input tensor. Must be 3D.
        dg: Rotation degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.pi/180 * dg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def get_patch(x,offset_h,h,offset_w,w):
    """
        get a patch or sub-image or ROI or sub matrix of size h x w
        starting from offset_h and offset_w.

        args:
            - x (numpy array): image or matrix (x.shape = h,w,c channels last)
    """
    return x[offset_h:offset_h+h,offset_w:offset_w+w]

def get_random_X_Y_patches(x,y,h,w):
    """
        Get same random patches for X and Y.
    """
    offset_h = np.random.randint(h)
    offset_w = np.random.randint(w)
    return get_patch(x,offset_h,h,offset_w,w),get_patch(y,offset_h,h,offset_w,w)

def get_numpy_from_path(image_path, convert='Label'):
    """
        args:
            image_path
            convert (String) : 'Label' converts to boolean labels
    """
    if convert=='Label':
        data = np.array(load_img(image_path).convert('1'))
        return np.memmap( data.reshape(list(data.shape)+[1]), dtype='float32' )
    if convert=='Train':
        return np.memmap( np.array(load_img(image_path)), dtype='float32' )

def get_data_from_directory(directory, convert='Label'):
    files = sorted(os.listdir(directory))
    if len(files)<2:
        return None
    a_name = files[0]
    a = get_numpy_from_path(os.path.join(directory,a_name), convert=convert)
    b_name = files[1]
    b = get_numpy_from_path(os.path.join(directory,b_name), convert=convert)
    stack = np.stack((a, b))
    for file_name in files[2:]:
        c = get_numpy_from_path(os.path.join(directory,file_name), \
                                                    convert=convert)
        stack = np.concatenate((stack,c.reshape([1]+list(c.shape))), axis=0)
    return stack

def to_image(array):
    mx,mn = np.max(array),np.min(array)
    array = np.array(np.floor(((array - mn)/(mx-mn))*255),dtype=np.uint8)
    return array

def save_image(array_image,image_path):
    """
        Save numpy array as image
    """
    i = PIL.Image.fromarray(to_image(array_image))
    i.save(image_path)

def main():
    path = '../CD_Dataset/train_y/03_bin.png'
    path_store = "../CD_Dataset/train_w/03_w.png"
    image = PIL.Image.open(path)
    data = np.array(image)/255.
    plt.imshow(data)
    plt.show()
    morphow = morphological_weights(data)
    morphow = (1-data)*morphow
    save_image(morphow, path_store)
    plt.imshow(data+morphow)
    plt.show()

if __name__ == '__main__':
    main()
