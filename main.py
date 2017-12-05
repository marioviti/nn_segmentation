from datasets.CD_Dataset import CD_Dataset
from models.Unet import Unet

import matplotlib.pyplot as plt
import numpy as np

from utils import train, predict_patch, show_image_x_y, predict_and_show

################################################################################
################################## PARAMETERS ##################################
################################################################################
INPUT_PATCH_SIZE = [320,320]
INPUT_CHANNELS = [3]
EPOCHS = 2
N_PATCH_BATCH = 5
MODEL_NAME = "Unet"
################################################################################

def main():

    # Load data
    dataset_path = './CD_Dataset'
    dataset = CD_Dataset( dataset_path, download=True )

    # Define Model
    inputs_shape = INPUT_PATCH_SIZE + INPUT_CHANNELS
    unet = Unet( inputs_shape )
    # Ouput size is defined by the model
    outputs_shape= unet.get_model_output_shape()

    # Set sizes to the datagenerator
    dataset.set_sizes( inputs_shape, outputs_shape )

    # Compute preprocessing parameters
    dataset.fit()

    # Train the model
    unet.load_model(name=MODEL_NAME)
    scores,losses = train( unet, dataset, epochs=EPOCHS, n_batch=N_PATCH_BATCH)
    print ('LOSS')
    print (losses)
    print ('SCORE')
    print (scores)
    unet.save_model(name=MODEL_NAME)
    x = dataset.get_example()
    y = unet.predict_and_stich(x)
    print(x.shape,y.shape)
    show_image_x_y(x,y)
    predict_and_show(unet,dataset)

if __name__ == '__main__':
    main()
