from matplotlib import pyplot as plt
import numpy as np

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

# from categorical for a batch patch of size
# N_batch,H,W,Classes, inverse operation of to_categorical
def Y_to_image(Y):
    return np.argmax(Y,axis=2)

def predictBatchXYandShow(model, dataset, n_batch=25):
    h,w,_ = model.input_shape
    x, y = dataset.sample_X_Y_patch_batch([h,w],n_batch=n_batch,train=True)
    y_hat = model.predict(x)
    model.evaulate(x,y)
    loss, metric = model.score
    x = crop_receptive(x,y_hat.shape[1:])
    y = crop_receptive(y,y_hat.shape[1:])
    fig = plt.figure()
    for i in range(n_batch):
        x_image = to_image(x[i])
        y_image = Y_to_image(y[i])
        y_image_hat = Y_to_image(y_hat[i])
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
