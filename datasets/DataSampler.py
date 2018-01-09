from utils import batch_patches, images_to_numpy, to_categorical,\
                  from_categorical, refuse_batch, images_random_rotate

import numpy as np

class DataSampler():
    def __init__(self, dataholder, num_classes=2):
        self.num_classes = num_classes
        self.dataholder = dataholder

    # return total number of patches from_images*batch_size
    def get_batch_patches(self, h, w, from_images=3, batch_size=4,
                            rotation_range=20,
                            as_is=False,
                            acceptance=True,
                            y_index = 1,
                            apply_w=False,**kwargs):
        """
            return cat_batches: [Xs,Ys], [Xs,Ys,Ws]
        """
        images = images_random_rotate(images_to_numpy(self.dataholder.get_next(**kwargs)))
        cat_batches = batch_patches(images,batch_size,h,w)
        N_batches = len(cat_batches)
        max_iterations = 30
        for i in range(1,from_images):
            iterations = 0
            not_accepted = True
            images = images_random_rotate(images_to_numpy(self.dataholder.get_next(**kwargs)))
            batches = batch_patches(images,batch_size,h,w)
            while(not_accepted and acceptance and iterations<max_iterations):
                batches = batch_patches(images,batch_size,h,w)
                y_batch = batches[y_index]
                not_accepted = refuse_batch(y_batch)
                iterations += 1
            for k in range(N_batches):
                cat_batches[k] = np.concatenate((cat_batches[k],batches[k]),
                                                                    axis=0)
        if as_is:
            return cat_batches
        else:
            return self.process_batches(cat_batches,y_index=y_index)

    def process_batches(self,batches,y_index=1):
        batches
        batches[y_index] = to_categorical(batches[y_index], num_classes=self.num_classes)
        return batches
