from datasets import CD_Dataset
from utils import show_Images, to_image
import numpy as np

def main():
    DATASET_PATH = './CD_Dataset'
    dataset = CD_Dataset( path=DATASET_PATH, download=True, fit=False )
    h,w = 1000,1000
    n_batch = 2
    x, w, y = dataset.get_X_W_Y_patch_batch([h,w],n_batch=n_batch)
    print(y)

    show_Images(x, y, W=w)

if __name__ == '__main__':
    main()
