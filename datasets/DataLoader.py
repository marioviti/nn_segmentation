from PIL import Image
import os

class DataLoader():
    def __init__(self, path, train_x_path = "train_x",
                             train_w_path = "train_w",
                             train_y_path = "train_y",
                             eval_x_path = "eval_x", eval_y_path = "eval_y" ):
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
        self.eval_y_path = os.path.join(path,eval_y_path)

        self.train_x_directory_list = sorted(os.listdir(self.train_x_path))
        self.train_w_directory_list = sorted(os.listdir(self.train_w_path))
        self.train_y_directory_list = sorted(os.listdir(self.train_y_path))
        self.eval_x_directory_list = sorted(os.listdir(self.eval_x_path))
        self.eval_y_directory_list = sorted(os.listdir(self.eval_y_path))

        self.train_x = [Image.open(os.path.join(self.train_x_path,x))\
                                        for x in self.train_x_directory_list]
        self.train_w = [Image.open(\
                            os.path.join(self.train_w_path,w)).convert('L')\
                                        for w in self.train_w_directory_list]
        self.train_y = [Image.open(\
                            os.path.join(self.train_y_path,y)).convert('L')\
                                        for y in self.train_y_directory_list]
        self.eval_x = [Image.open(os.path.join(self.eval_x_path,x))\
                                        for x in self.eval_x_directory_list ]
        self.eval_y = [Image.open(\
                            os.path.join(self.eval_y_path,y)).convert('L')\
                                        for y in self.eval_y_directory_list]

    def get_data(self):
        """
            return train_w, train_x, train_y, eval_x, eval_y as PIL images lists
        """
        return  self.train_w,\
                self.train_x, self.train_y,\
                self.eval_x, self.eval_y
