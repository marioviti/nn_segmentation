from datasets import DataManager
from utility import train
from models import Unet

DATASET_PATH = './CD_Dataset'
MODEL_PATH_NAME = './Unet'
INPUT_PATCH_SIZE = [350,350]
INPUT_CHANNELS = [3]
OUTPUT_CHANNELS = [2]
EPOCHS = 20

def main():
    datamanager = DataManager(DATASET_PATH)
    model_input_path = INPUT_PATCH_SIZE + INPUT_CHANNELS
    unet = Unet(model_input_path)
    train(unet, datamanager, epochs=EPOCHS)
    unet.save_model(MODEL_PATH_NAME)

if __name__ == '__main__':
    main()
