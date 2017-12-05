from data_grab import dwuzp

from keras.preprocessing.image import random_rotation ,apply_transform, random_zoom, load_img
from keras.utils.np_utils import to_categorical

import numpy as np
import os

import PIL
from preprocessing import *


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

    def get_data_all(self):
        """
            return train_x, train_y, eval_x, eval_y as PIL images lists
        """
        return self.train_x, self.train_y, self.eval_x, self.eval_y


class Data_Generator():
    def __init__(self, train_x, train_y, eval_x=None, eval_y=None):
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
        self.example_id = 0

    def fit(self):
        """
            preprocessing pre-step:
                calculate mean, std featurewise (out of memory).
        """
        self.N = len(self.train_x)
        N = self.N
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

    def get_Y(self, index=0, eval_=True):
        """
            args:
        """
        y = np.array(self.eval_y[index] if eval_ else self.train_y[index])
        return y

    def get_X(self, index=0, fitted=True, eval_=True):
        """
            args:
                -fitted (boolean) : if fitted the example preprocessed
        """
        x = np.array(self.eval_x[index] if eval_ else self.train_x[index])
        if self.fitted and fitted:
            x = (x/255.0 - self.mean_features)/(self.std_features+1e-10)
        return x

    def get_X_Y_patches(self,x,y,h,w):
        """
            get same random patch and apply mean std featurewise normalization.
        """
        cy = 2
        patch_ax, patch_ay = get_random_X_Y_patches(x,y,h,w)
        patch_ax, patch_ay = patch_ax/255., patch_ay/255
        patch_ay = to_categorical(patch_ay, num_classes=cy)
        _,_,cx = patch_ax.shape
        _,_,cy = patch_ay.shape
        if self.fitted:
            patch_ax = (patch_ax - self.mean_features)/(self.std_features+1e-10)
        return patch_ax.reshape([1,h,w,cx]), patch_ay.reshape([1,h,w,cy])

    def get_X_Y_patch_batch(self, patch_size, n_batch=10, shuffle=True, same=False):
        """
            args:
                patch_size (tuple) : hx, wx height and width window patch sizes
        """
        assert(n_batch>0 and n_batch<self.N )
        N = self.N
        hx,wx = patch_size
        self.example_id = (self.example_id+1)%N if not same else self.example_id
        idx = np.random.randint(N) if shuffle else self.example_id
        datax, datay = np.array(self.train_x[idx]), np.array(self.train_y[idx])
        batch_x, batch_y  = self.get_X_Y_patches( datax,datay,hx,wx )
        for i in range(1,n_batch):
            patch_x, patch_y = self.get_X_Y_patches( datax,datay,hx,wx )
            batch_x = np.concatenate( (batch_x,patch_x), axis=0 )
            batch_y = np.concatenate( (batch_y,patch_y), axis=0 )
        return batch_x,batch_y

class CD_Dataset():
    def __init__( self, path="../CD_Dataset", download=False, fit=True ):
        if (not os.path.exists(path)) and download:
            print('Downloading CD_Dataset')
            dwuzp()
        self.loader = Dataset_Loader(path)
        x_train, y_train, x_eval, y_eval = self.loader.get_data_all()
        self.generator = Data_Generator( x_train, y_train, x_eval, y_eval )
        if fit:
            self.fit()

    def fit(self):
        self.generator.fit()

    def get_X_Y_patch_batch( self, patch_size,  shuffle=True, n_batch=10, \
                             same=False ):
        return self.generator.get_X_Y_patch_batch( patch_size, shuffle=shuffle,\
                                               n_batch=n_batch, same=same )

    def get_X( self, index=0, fitted=True, eval_=False ):
        return self.generator.get_X( index=index, fitted=fitted, eval_=eval_ )

    def get_Y( self, index=0, eval_=False ):
        return self.generator.get_Y( index=index, eval_=eval_ )
