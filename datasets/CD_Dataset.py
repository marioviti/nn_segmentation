import os
import numpy as np
import PIL
from keras.preprocessing.image import random_rotation ,apply_transform, random_zoom, load_img
from preprocessing import *
from data_grab import dwuzp

class Dataset_Loader():
    def __init__(self, path, train_x_path = "train_x", train_y_path = "train_y"\
                        ,eval_x_path = "eval_x", eval_y_path = "eval_y" ):
        """
            path/
                train_x_path/
                train_y_path/
                eval_x_path/
                eval_y_path/

            Using PIL.Image out of memory pointers for images

            args:
                - path (string) : Path to the dataset image
        """
        self.path = path
        self.train_x_path = os.path.join(path,train_x_path)
        self.train_y_path = os.path.join(path,train_y_path)
        self.eval_x_path = os.path.join(path,eval_x_path)
        self.eval_y_path = os.path.join(path,eval_y_path)

        self.train_x_directory_list = sorted(os.listdir(self.train_x_path))
        self.train_y_directory_list = sorted(os.listdir(self.train_y_path))
        self.eval_x_directory_list = sorted(os.listdir(self.eval_x_path))
        self.eval_y_directory_list = sorted(os.listdir(self.eval_y_path))

        self.train_x = [PIL.Image.open(os.path.join(self.train_x_path,x))\
                                        for x in self.train_x_directory_list]
        self.train_y = [PIL.Image.open(\
                            os.path.join(self.train_y_path,y)).convert('L')\
                                        for y in self.train_y_directory_list]
        self.eval_x = [PIL.Image.open(os.path.join(self.eval_x_path,x))\
                                        for x in self.eval_x_directory_list ]
        self.eval_y = [PIL.Image.open(\
                            os.path.join(self.eval_y_path,y)).convert('L')\
                                        for y in self.eval_y_directory_list]

    def get_images(self):
        """
            return train_x, train_y, eval_x, eval_y as PIL images lists
        """
        return self.train_x, self.train_y, self.eval_x, self.eval_y


class Data_Generator():
    def __init__(self, model_input_shape, model_output_shape, train_x, train_y, eval_x=None, eval_y=None):
        self.model_input_shape = model_input_shape
        self.model_output_shape = model_output_shape
        self.train_x = train_x
        self.train_y = train_y
        self.eval_x = eval_x
        self.eval_y = eval_y
        self.next_train_x = None
        self.next_train_y = None
        self.next_eval_x = None
        self.next_eval_y = None
        self.mean_features = None
        self.std_features = None
        self.fitted = False
        self.curr_idx = 0

    def fit(self):
        """
            preprocessing pre-step:
                calculate mean, std featurewise (out of memory).
        """
        N = len(self.train_x)
        for idx in range(N):
            datax = np.array(self.train_x[idx])/255.
            H,W = datax.shape[0:2]
            if self.mean_features is None:
                self.mean_features = np.sum(datax,axis=(0,1))/(N*H*W*1.0)
            else:
                self.mean_features += np.sum(datax,axis=(0,1))/(N*H*W*1.0)
        for idx in range(N):
            datax = np.array(self.train_x[idx])/255.
            H,W = datax.shape[0:2]
            datax = datax - self.mean_features
            datax = datax**2
            if self.std_features is None:
                self.std_features = np.sum(datax,axis=(0,1))/(N*H*W*1.0-1.0)
            else:
                self.std_features += np.sum(datax,axis=(0,1))/(N*H*W*1.0-1.0)
        self.fitted = True

    def get_random_fit_X_Y_patches(self,x,y,h,w):
        """
            get same random patch and apply mean std featurewise normalization.
        """
        ax, ay = get_X_Y_patches(x,y,h,w)
        chns = self.model_input_shape[2]
        lbls = self.model_output_shape[2]
        if self.fitted:
            ax, ay = ax/255., ay/255.
            ax = (ax - self.mean_features)/(self.std_features+1e-10)
        return ax.reshape([1,h,w,chns]),\
                    crop_receptive_field(ay.reshape([h,w,lbls]),self.model_output_shape)

    def get_patch_batch(self, h=None, w=None, shuffle=True,\
                                                n_batch=10, same_image=False):
        assert(n_batch>0)
        N = len(self.train_x)
        if h == None:
            h,w,_ = self.model_input_shape
        self.curr_idx = (self.curr_idx+1)%N if not same_image else self.curr_idx
        idx = np.random.randint(N) if shuffle else self.curr_idx
        datax, datay = np.array(self.train_x[idx]), np.array(self.train_y[idx])
        batch_x,batch_y  = self.get_random_fit_X_Y_patches(datax,datay,h,w)
        for i in range(1,n_batch):
            cx, cy = self.get_random_fit_X_Y_patches(datax,datay,h,w)
            batch_x = np.concatenate((batch_x,cx), axis=0)
            batch_y = np.concatenate((batch_y,cy), axis=0)
        return batch_x,batch_y

class CD_Dataset():

    def __init__(self,path="../CD_Dataset",download=False):
        if (not os.path.exists(path)) and download:
            print('Downloading CD_Dataset')
            dwuzp()
        self.loader = Dataset_Loader(path)

    def set_sizes(self, model_input_shape, model_output_shape):
        x_train, y_train, x_eval, y_eval = self.loader.get_images()
        self.generator = Data_Generator(model_input_shape, model_output_shape,\
                                            x_train, y_train, x_eval, y_eval)
    def fit(self):
        self.generator.fit()

    def get_patch_batch(self, shuffle=True, n_batch=10, same_image=False):
        return self.generator.get_patch_batch(shuffle=shuffle, n_batch=n_batch,\
                                                        same_image=same_image)
