from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.set_image_data_format('channels_last')
import os

def main():
    train_x_path = '../CD_Dataset/train_x'
    train_y_path = '../CD_Dataset/train_y'
    train_w_path = '../CD_Dataset/train_w'

    x_datagen = ImageDataGenerator()
    y_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    horizontal_flip=True)
    seed = 1
    x_generator = x_datagen.flow_from_directory(
        train_x_path, seed=seed)


    #train_generator = zip(x_generator, y_generator)



if __name__ == '__main__':
    main()
