from DataLoader import DataLoader
from DataHolder import DataHolder
from DataSampler import DataSampler
from DataShower import DataShower
import os
from data_grab import dwuzp

class DataManager():
    def __init__( self, path="../CD_Dataset", download=False , num_classes=2):
        """
          Args:
              - path : to dataset main folder
              - download : if set download from url
        """
        if (not os.path.exists(path)) and download:
            print( 'Downloading CD_Dataset' )
            dwuzp()
        self.dataloader = DataLoader(path)

        train_w, train_x, train_y, eval_x, eval_y = self.dataloader.get_data()
        self.train_dataholder = DataHolder(train_x, train_y, W= train_w)
        self.eval_dataholder = DataHolder(eval_x, eval_y)
        self.train_datasampler = DataSampler(self.train_dataholder,num_classes=num_classes)
        self.eval_datasampler = DataSampler(self.train_dataholder,num_classes=num_classes)
        self.datashower = DataShower()

    def get_eval_batch_patches( self, *args, **kwargs ):
        return self.eval_datasampler.get_batch_patches( *args, **kwargs )

    def get_train_batch_patches( self, *args, **kwargs ):
        btch = self.train_datasampler.get_batch_patches( *args, **kwargs )
        print(len(btch))
        return btch[0:2]

    def show_train_batch_patches( self, *args, **kwargs ):
        batches = self.train_datasampler.get_batch_patches( *args, **kwargs )
        self.datashower.show_batches(batches)
