import argparse

from datasets import CD_Dataset
from models import Unet

import matplotlib.pyplot as plt
import numpy as np

import string
import time

import random
from utils import train, predict_patch, show_image_x_y,\
                  predict_and_show, show_y, predictBatchXYandShow

np.random.seed(int((time.time()*1e6)%1e6))

################################################################################
########################## SESSION ARGUMENTS ###################################
################################################################################

global MODEL_PATH_NAME
global MODEL_TRAINING_SESSION
global MODEL_TESTING_SESSION
global LOAD_MODEL
global EPOCHS
global N_PATCH_BATCH

DATASET_PATH = './CD_Dataset'
EPOCHS = 50
N_PATCH_BATCH = 5

################################################################################
############################ MODEL PARAMETERS ##################################
################################################################################

global INPUT_PATCH_SIZE
global INPUT_CHANNELS
global OUTPUT_CHANNELS

INPUT_PATCH_SIZE = [350,350]
INPUT_CHANNELS = [3]
OUTPUT_CHANNELS = [2]

################################################################################

def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def main():
    global MODEL_PATH_NAME
    global MODEL_TRAINING_SESSION
    global MODEL_TESTING_SESSION
    global LOAD_MODEL
    global EPOCHS
    global N_PATCH_BATCH
    global INPUT_PATCH_SIZE
    global INPUT_CHANNELS
    global OUTPUT_CHANNELS

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-trn","--training",
                        help="specify if trainig session", action="store_true")
    parser.add_argument("-tst","--testing",
                        help="specify if testing session", action="store_true")
    parser.add_argument("-epc","--epochs",
                        help="number of epochs", type=int, default=10)
    parser.add_argument("-btc","--n_batch",
                        help="number of examples per batch", type=int, default=5)
    parser.add_argument("-ldmd","--load_model",
                        help="specify model path/name, the script will load the\
                            path/name.h5 and path/name.json files")

    # Passing arguments to session
    args = parser.parse_args()
    MODEL_TRAINING_SESSION = args.training
    EPOCHS = args.epochs
    N_PATCH_BATCH = args.n_batch
    MODEL_TESTING_SESSION = args.testing
    if args.load_model is not None:
        MODEL_PATH_NAME = args.load_model
        LOAD_MODEL = True
    else:
        MODEL_PATH_NAME = id_generator()
        print("no model selected creating a new one: "+MODEL_PATH_NAME)
        LOAD_MODEL = False
    run_model()

def run_model():
    # Running the model
    dataset = CD_Dataset( path=DATASET_PATH, download=True, fit=True )

    model_input_path = INPUT_PATCH_SIZE + INPUT_CHANNELS
    unet = Unet(model_input_path)

    if LOAD_MODEL:
        print("loading model " + MODEL_PATH_NAME + " from disk.")
        unet.load_model(MODEL_PATH_NAME)

    if MODEL_TRAINING_SESSION:
        print("trainig model")
        train(unet,dataset, epochs=EPOCHS, n_batch=N_PATCH_BATCH)
        print("saving model " + MODEL_PATH_NAME + " to disk.")
        unet.save_model(MODEL_PATH_NAME)

    elif MODEL_TESTING_SESSION:
        print('testing model')
        predictBatchXYandShow(unet, dataset, n_batch=N_PATCH_BATCH)
    else:
        print("no mode selected: add option --help to see mode options.")

if __name__ == '__main__':
    main()
