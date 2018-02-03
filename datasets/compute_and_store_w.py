from CD_Dataset import Dataset_Loader
from preprocessing import morphological_weights, save_image
import numpy as np
import os

"""
Use this script to compute morphological distance within cells.
set the paths and run it.
"""

default_path = "../CD_Dataset"
default_train_output_path = "train_w"
default_eval_output_path = "eval_w"
w_suffix = "_w"
extension = '.png'

def compute_and_store(image, path_store):
    data = np.array(image)/255.
    morphow = morphological_weights(data)
    morphow = (1-data)*morphow
    save_image(morphow, path_store)

def main():
    data_loader = Dataset_Loader(default_path)

    train_y_directory_list = data_loader.train_y_directory_list
    train_y = data_loader.train_y

    eval_y_directory_list = data_loader.eval_y_directory_list
    eval_y = data_loader.eval_y

    train_path = os.path.join(default_path,default_train_output_path)
    eval_path = os.path.join(default_path,default_eval_output_path)

    os.mkdir(train_path)
    os.mkdir(eval_path)
    for i in range(len(train_y_directory_list)):
        name = train_y_directory_list[i][-10:-8] + w_suffix + extension
        path_store = os.path.join(train_path,name)
        image = train_y[i]
        compute_and_store(image,path_store)
    for i in range(len(eval_y_directory_list)):
        name = eval_y_directory_list[i][-10:-8] + w_suffix + extension
        path_store = os.path.join(eval_path,name)
        image = eval_y[i]
        compute_and_store(image,path_store)


if __name__ == '__main__':
    main()
