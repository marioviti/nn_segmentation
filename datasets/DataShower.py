from matplotlib import pyplot as plt
import numpy as np

def to_image(x):
    print(x.shape)
    mn,mx = np.min(x), np.max(x)
    return (x - mn)/float(mx-mn)

class DataShower():
    def __init__(self):
        self.figure = plt.figure()

    def show_batches(self, batches, batch_titles=None):
        fig = self.figure
        n_batches = len(batches)
        n_patches = batches[0].shape[0]
        for i in range(n_patches):
            for j in range(n_batches):
                a=fig.add_subplot(n_patches,n_batches,i*n_batches+j+1)
                image_dims = len(batches[j][i].shape)
                if image_dims==2:
                    plt.imshow(batches[j][i]*255)
                else:
                    plt.imshow(batches[j][i])
                if batch_titles is not None:
                    a.set_title("{}_example_{}".format(batch_titles[j],i))
        plt.show()
