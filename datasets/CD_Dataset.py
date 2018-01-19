from data_grab import dwuzp

from keras.preprocessing.image import random_rotation ,apply_transform, random_zoom, load_img
from keras.utils.np_utils import to_categorical

import numpy as np
import time
import os
import PIL
from utils import sample_patches, refuse_batch

np.random.seed(int((time.time()*1e6)%1e6))

class Dataset_Loader():
    def __init__(self, path, train_x_path = "train_x",
                             train_w_path = "train_w",
                             train_y_path = "train_y",
                             eval_x_path = "eval_x", 
                             eval_w_path = "eval_w", 
                             eval_y_path = "eval_y" ):
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
        self.train_w_path = os.path.join(path,train_w_path)
        self.train_y_path = os.path.join(path,train_y_path)
        self.eval_x_path = os.path.join(path,eval_x_path)
        self.eval_w_path = os.path.join(path,eval_w_path)
        self.eval_y_path = os.path.join(path,eval_y_path)

        self.train_x_directory_list = sorted(os.listdir(self.train_x_path))
        self.train_w_directory_list = sorted(os.listdir(self.train_w_path))
        self.train_y_directory_list = sorted(os.listdir(self.train_y_path))
        self.eval_x_directory_list = sorted(os.listdir(self.eval_x_path))
        self.eval_w_directory_list = sorted(os.listdir(self.eval_w_path))
        self.eval_y_directory_list = sorted(os.listdir(self.eval_y_path))

        self.train_x = [PIL.Image.open(os.path.join(self.train_x_path,x))\
                                        for x in self.train_x_directory_list]
        self.train_w = [PIL.Image.open(\
                            os.path.join(self.train_w_path,w)).convert('L')\
                                        for w in self.train_w_directory_list]
        self.train_y = [PIL.Image.open(\
                            os.path.join(self.train_y_path,y)).convert('L')\
                                        for y in self.train_y_directory_list]
        self.eval_x = [PIL.Image.open(os.path.join(self.eval_x_path,x))\
                                        for x in self.eval_x_directory_list ]
        self.eval_w = [PIL.Image.open(\
                            os.path.join(self.eval_w_path,w)).convert('L')\
                                        for w in self.eval_w_directory_list]
        self.eval_y = [PIL.Image.open(\
                            os.path.join(self.eval_y_path,y)).convert('L')\
                                        for y in self.eval_y_directory_list]

    def get_data_all(self):
        """
            return train_x, train_w, train_y, eval_x, eval_y as PIL images lists
        """
        return  self.train_x, self.train_w, self.train_y, \
                self.eval_x, self.eval_w, self.eval_y


class Data_Sampler():
    def __init__(self, train_x, train_w, train_y,
                       eval_x=None, eval_w=None, eval_y=None, 
                       num_classes=2):
        """
            Args:
                - train_x : (PIL.Image pointers list)
                ...
                - eval_y : (PIL.Image pointers list)
                - classes : (int) number of classes for categorical encoding.
        """
        self.train_x = train_x
        self.train_w = train_w
        self.train_y = train_y
        self.eval_x = eval_x
        self.eval_w = eval_w
        self.eval_y = eval_y
        self.num_examples = len(self.train_x)
        self.eval_num_examples = len(self.eval_x)
        self.num_classes = num_classes
        self.mean_features = None
        self.std_features = None
        self.fitted = False
        self.curr_example_id = 0
        self.eval_curr_example_id = 0

    @property
    def mean_features(self):
        return self.mean_features

    @mean_features.setter
    def mean_features(self, v):
        self.mean_features = v

    @property
    def std_features(self):
        return self.mean_features

    @std_features.setter
    def std_features(self, v):
        self.std_features = v

    def get_next_index(self,train=True):
        if train:
            N = self.num_examples
            self.curr_example_id = (self.curr_example_id+1)%N
            return self.curr_example_id
        else:
            N = self.eval_num_examples
            self.eval_curr_example_id = (self.eval_curr_example_id+1)%N
            return self.eval_curr_example_id

    def get_random_index(self,train=True):
        N = self.num_examples if train else self.eval_num_examples
        return np.random.randint(N)

    def get_curr_index(self,train=True):
        return self.curr_example_id if train else self.eval_curr_example_id


    def fit(self):
        """
            preprocessing pre-step:
                calculate mean, std featurewise (out of memory).
        """
        N = self.num_examples
        datax = np.array(self.train_x[0])/255.
        H,W = datax.shape[0:2]
        self.mean_features = np.sum(datax,axis=(0,1))
        for idx in range(1,N):
            datax = np.array(self.train_x[idx])/255.
            self.mean_features += np.sum(datax,axis=(0,1))
        self.mean_features /= (N*H*W*1.0)
        datax = np.array(self.train_x[0])/255.
        H,W = datax.shape[0:2]
        datax = datax - self.mean_features
        datax = datax**2
        self.std_features = np.sum(datax,axis=(0,1))
        for idx in range(N):
            datax = np.array(self.train_x[idx])/255.
            H,W = datax.shape[0:2]
            datax = datax - self.mean_features
            datax = datax**2
            self.std_features += np.sum(datax,axis=(0,1))
        self.std_features /= (N*H*W*1.0-1.0)
        print("mean_features: ",self.mean_features)
        print("std_features: ",self.std_features)
        self.fitted = True

    def get_X_Y_W(self, index=None, shuffle=True, train=True, rotated=True):
        idx = self.get_curr_index()
        idx = self.get_random_index(train=train) if shuffle else idx
        idx = index if index is not None else idx
        X = self.train_x[idx] if train else self.eval_x[idx]
        Y = self.train_y[idx] if train else self.eval_y[idx]
        W = self.train_w[idx] if train else self.eval_w[idx]
        X = np.array(X)/255.
        W = np.array(W)/255.
        Y = np.array(Y)/255            
        if self.fitted:
            X = (X - self.mean_features)/(self.std_features+1e-10)
        cy = self.num_classes
        Y = to_categorical(Y, num_classes=cy)
        return X,Y,W

    def sample_X_Y_W_patches(self,patch_size,X,Y,W,fit=True,offsets=[None,None]):
        """
            get same random patch and apply mean std featurewise normalization.
        """
        cy = self.num_classes
        h,w = patch_size
        offset_h,offset_w = offsets
        
        refused = True
        i = 0
        while(refused and i<10):
            patch_ax, patch_ay, patch_aw = sample_patches([X,Y,W],h,w,offset_h=offset_h,offset_w=offset_w)
            patch_ax, patch_ay, patch_aw = patch_ax/255., patch_ay/255, patch_aw/255.
            refused = refuse_batch(patch_ay)
            i += 1
  
        patch_ay = to_categorical(patch_ay, num_classes=cy)
            
        if self.fitted and fit:
            patch_ax = (patch_ax - self.mean_features)/(self.std_features+1e-10)
        return np.expand_dims(patch_ax, axis=0), np.expand_dims(patch_ay, axis=0), np.expand_dims(patch_aw, axis=0)

    def sample_X_Y_W_patch_batch(self, patch_size, n_batch=10,
                                     offsets = [None,None],
                                     train=True, fit=True,
                                     shuffle=True, same=False):
        """
            args:
                patch_size (tuple) : h, w height and width window patch sizes
        """
        assert(n_batch>0)
        idx = self.get_curr_index(train=train) if same else (self.get_random_index(train=train) \
                                        if shuffle else self.get_next_index(train=train))
        
        datax = np.array(self.train_x[idx] if train else self.eval_x[idx] )
        datay = np.array(self.train_y[idx] if train else self.eval_y[idx] )
        dataw = np.array(self.train_w[idx] if train else self.eval_w[idx] )

        batch_x, batch_y, batch_w  = self.sample_X_Y_W_patches(patch_size,datax,datay,dataw,fit=fit,offsets=offsets)
        for i in range(1,n_batch):
            patch_x, patch_y, patch_w = self.sample_X_Y_W_patches(patch_size,datax,datay,dataw,fit=fit,offsets=offsets)
            batch_x = np.concatenate( (batch_x,patch_x), axis=0 )
            batch_w = np.concatenate( (batch_w,patch_w), axis=0 )
            batch_y = np.concatenate( (batch_y,patch_y), axis=0 )
        return batch_x, batch_y, batch_w


class CD_Dataset():
    def __init__( self, path="../CD_Dataset",
                  download=False, fit=False, num_classes=2 ):
        """
          Args:
              - path : to dataset main folder
              - download : if set download from url
              - classes : (int) number of classes for categorical encoding.
              - fit : preprocess fitting (compute paramenters)
        """
        if (not os.path.exists(path)) and download:
            print( 'Downloading CD_Dataset' )
            dwuzp()
        self.loader = Dataset_Loader(path)
        train_x, train_w, train_y, eval_x, eval_w, eval_y = self.loader.get_data_all()
        self.sampler = Data_Sampler( train_x, train_w, train_y,
                                     eval_x=eval_x, eval_w=eval_w, eval_y=eval_y, num_classes=num_classes )
        if fit:
            self.sampler.fit()
    
    def mean_features(self):
        return self.sampler.mean_features if self.sampler.fit else None

    def std_features(self):
        return self.sampler.std_features if self.sampler.fit else None
    
    def sample_X_Y_W_patch_batch( self, patch_size, **kwargs ):
        return self.sampler.sample_X_Y_W_patch_batch( patch_size, **kwargs )

    def get_X_Y_W(self,*args,**kwargs):
        return self.sampler.get_X_Y_W(*args,**kwargs)

    def sample_XY_patch_at(self,patch_size,offsets,train=True,same=False):
        return self.sampler.sample_X_Y_patch_batch( patch_size, n_batch=1,
                                                     offsets = offsets,
                                                     YW=False, train=train,
                                                     shuffle=False, same=same)
