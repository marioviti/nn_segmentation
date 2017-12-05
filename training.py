from datasets import CD_Dataset
from models import Unet

import matplotlib.pyplot as plt
import numpy as np

from utils import train, predict_patch, show_image_x_y, predict_and_show, show_y

################################################################################
################################## PARAMETERS ##################################
################################################################################
INPUT_PATCH_SIZE = [350,350]
INPUT_CHANNELS = [3]
OUTPUT_CHANNELS = 2
EPOCHS = 10
N_PATCH_BATCH = 10
MODEL_NAME = "Unet"
################################################################################

def main():
    input_shape = INPUT_PATCH_SIZE + INPUT_CHANNELS

    # Load data
    dataset_path = './CD_Dataset'
    dataset = CD_Dataset( dataset_path, download=True )
    unet = Unet(input_shape)
    unet.load_model(MODEL_NAME)
    train(unet,dataset, epochs=EPOCHS, n_batch=N_PATCH_BATCH)
    unet.save_model(MODEL_NAME)

if __name__ == '__main__':
    main()
