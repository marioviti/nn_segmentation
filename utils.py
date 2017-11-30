from matplotlib import pyplot as plt
import numpy as np

def train(model,dataset, epochs=10, n_batch=5):
    x_train,y_train = None,None
    scores = np.zeros(epochs)
    losses = np.zeros(epochs)
    for i in range(epochs):
        x_train,y_train = dataset.get_patch_batch(n_batch=n_batch)
        model.fit(x_train,y_train)
        model.evaulate(x_train,y_train)
        loss,score = model.score
        scores[i] = score
        losses[i] = loss
    return scores,losses

def predict_patch(model,x):
    y_hat = model.predict(x)
    return y_hat[0]

def predict(model,x):
    """
        args:
            model (Unet) : keras implementation
            x (numpy array) : input image to segnmnt
    """
    full_size = x.shape
    y = np.zeros(full_size)
    input_shape = model.input_shape[1:]
    output_shape = model.output_shape


def to_image(x):
    mn,mx = np.min(x), np.max(x)
    return (x - mn)/float(mx-mn)

def show_image_x_y(x,y):
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    imgplot = plt.imshow(to_image(x))
    a.set_title('X')
    a=fig.add_subplot(1,2,2)
    imgplot = plt.imshow(y[:,:,0])
    imgplot.set_clim(0.0,0.7)
    a.set_title('Y')
    plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation='vertical')
    plt.tight_layout()
    plt.show()
