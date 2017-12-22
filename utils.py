from matplotlib import pyplot as plt
import numpy as np

def train(model, dataset, epochs=10, n_batch=5):
    h,w,_ = model.input_shape
    metrics = np.zeros(epochs)
    losses = np.zeros(epochs)
    print("starging trainig process")
    for i in range(epochs):
        print("main-epoch: "+str(i))
        x_train,y_train = dataset.get_X_Y_patch_batch([h,w],n_batch=n_batch)
        model.fit(x_train,y_train)
        model.evaulate(x_train,y_train)
        loss, metric = model.score
        metrics[i] = metric
        losses[i] = loss
    return metrics,losses

def to_image(x):
    mn,mx = np.min(x), np.max(x)
    return (x - mn)/float(mx-mn)

def crop_receptive(batch, crop_size):
    """
        Get a cropped batch to fit the perceptive field,
        the resulting output shape is n,hy,wy,cy.

        args:
            - batch (numpy array) y.shape : n,hx,wx,cy
            - crop_size (list) : hy,wy,cy
    """
    n,hx,wx,cy = batch.shape
    hy,wy,cy = crop_size
    dhq, dhr = (hx-hy)//2, (hx-hy)%2
    dwq, dwr = (wx-wy)//2, (wx-wy)%2
    return batch[:, dhq: hx - (dhq + dhr), dwq: wx - (dwq + dwr) ]

def predictBatchXYandShow(model, dataset, n_batch=25):
    h,w,_ = model.input_shape
    x, y = dataset.get_X_Y_patch_batch([h,w],n_batch=n_batch)
    y_hat = model.predict(x)
    model.evaulate(x,y)
    loss, metric = model.score
    x = crop_receptive(x,y_hat.shape[1:])
    y = crop_receptive(y,y_hat.shape[1:])
    fig = plt.figure()
    print("starging prediction process")
    for i in range(n_batch):
        x_image = to_image(x[i])
        y_image = dataset.Y_to_image(y[i])
        y_image_hat = dataset.Y_to_image(y_hat[i])
        a=fig.add_subplot(n_batch,3,i*3+1)
        imgplot = plt.imshow(x_image)
        a.set_title('X')
        a=fig.add_subplot(n_batch,3,i*3+2)
        imgplot = plt.imshow(y_image)
        a.set_title('Y')
        a=fig.add_subplot(n_batch,3,i*3+3)
        imgplot = plt.imshow(y_image_hat)
        a.set_title('Y_hat')
        imgplot.set_clim(0.0,0.7)
    plt.show()
    print(loss, metric)


def predict_batch(model, x):
    return predict(model,x)

def predict_and_show(model,dataset,n_batch=10):
    x_train,y_train = dataset.get_patch_batch(n_batch=n_batch)
    for i in range(n_batch):
        x = x_train[i:i+0]
        y_hat = predict(model,x)
        show_image_x_y(x,y_hat,y_train[i])

def predict_batch(model, x):
    return predict(model,x)

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


def show_y(y):
    imgplot = plt.imshow(y)
    return imgplot

def show_image_x_y(x,y,yt=None):
    if yt is None:
        fig = plt.figure()
        a=fig.add_subplot(1,2,1)
        imgplot = plt.imshow(to_image(x))
        a.set_title('X')
        a=fig.add_subplot(1,2,2)
        imgplot = plt.imshow(y[:,:,0])
        imgplot.set_clim(0.0,0.7)
        a.set_title('Y')
    else:
        fig = plt.figure()
        a=fig.add_subplot(1,3,1)
        imgplot = plt.imshow(to_image(x))
        a.set_title('X')
        a=fig.add_subplot(1,3,2)
        imgplot = plt.imshow(y[:,:,0])
        a=fig.add_subplot(1,3,3)
        imgplot = plt.imshow(yt[:,:,0])
        imgplot.set_clim(0.0,0.7)
        a.set_title('Y')

    plt.colorbar(ticks=[0.1,0.3,0.5,0.7], orientation='vertical')
    plt.tight_layout()
    plt.show()
