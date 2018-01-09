import numpy as np
import time

np.random.seed(int((time.time()*1e6)%1e6))

class DataHolder():
    def __init__(self, X, Y, W=None ):
        """
            Args:
                - X : (PIL.Image pointers list)
                ...
        """
        self.X = X
        self.W = W
        self.Y = Y
        self.num_examples = len(self.X)
        self.curr_id = 0

    # Get indexes
    def get_next_index(self):
        N = self.num_examples
        self.curr_id = (self.curr_example_id+1)%N
        return self.curr_example_id

    def get_random_index(self):
        N = self.num_examples
        return np.random.randint(N)

    def get_curr_index(self):
        return self.curr_id

    def get_next(self, shuffle=True, next=False, curr=False, train=True):
        idx = 0
        idx = self.get_curr_index() if curr else idx
        idx = self.get_next_index() if next else idx
        idx = self.get_random_index() if shuffle else idx
        if self.W is None:
            examples = [ self.X[idx], self.Y[idx] ]
        else:
            examples = [ self.X[idx], \
                         self.Y[idx], \
                         self.W[idx] ]
        return examples
