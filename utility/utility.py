from matplotlib import pyplot as plt
import numpy as np

def crop_receptive(batch, crop_size):
    """
        Get a cropped batch to fit the perceptive field,
        the resulting output shape is n,hy,wy,cy.

        args:
            - batch (numpy array) y.shape : n,hx,wx,cy
            - crop_size (list) : hy,wy
    """
    n,hx,wx,_ = batch.shape
    hy,wy = crop_size
    dhq, dhr = (hx-hy)//2, (hx-hy)%2
    dwq, dwr = (wx-wy)//2, (wx-wy)%2
    return batch[:, dhq: hx - (dhq + dhr), dwq: wx - (dwq + dwr) ]

def evaluate(model,dataset,n_batch=10):
    h,w,_ = model.inputs_shape[0]
    n_batch = n_batch/3
    x_1,y_1,w_2= dataset.sample_X_Y_W_patch_batch([h,w],n_batch=n_batch, train=True)
    x_2,y_2,w_2= dataset.sample_X_Y_W_patch_batch([h,w],n_batch=n_batch, train=True)
    x_ = np.concatenate( (x_1,x_2), axis=0 )
    y_ = np.concatenate( (y_1,y_2), axis=0 )
    x_1,y_1,w_1= dataset.sample_X_Y_W_patch_batch([h,w],n_batch=n_batch, train=True)
    x_ = np.concatenate( (x_1,x_), axis=0 )
    y_ = np.concatenate( (y_1,y_), axis=0 )

    y_hat = model.predict(x_)
    x_ = crop_receptive(x_,y_hat.shape[1:3])
    y_ = crop_receptive(y_,y_hat.shape[1:3])
    return x_,y_,y_hat

def train(model,dataset, epochs=10, n_batch=10, use_weights=False, W=10, just_train=True, rotate=False):
    train_metrics = []
    eval_metrics = []

    h,w,_ = model.inputs_shape[0]
    for i in range(epochs):
        print("=========== iteration {}/{} =============".format(i+1,epochs))

        x_train,y_train,w_train = dataset.sample_X_Y_W_patch_batch([h,w],n_batch=n_batch,rotate=rotate)
        if use_weights:
            y_train = combine_y_w(y_train,w_train*W)    
        train_history = model.fit(x_train,y_train)
        train_metric = train_history.history.values()
        
        if not just_train:
            x_eval1,y_eval1,w_eval1 = dataset.sample_X_Y_W_patch_batch([h,w],n_batch=n_batch,train=False)
            x_eval2,y_eval2,w_eval2 = dataset.sample_X_Y_W_patch_batch([h,w],n_batch=n_batch,train=False)

            x_eval = np.concatenate( (x_eval1,x_eval2), axis=0 )
            y_eval = np.concatenate( (y_eval1,y_eval2), axis=0 )
            w_eval = np.concatenate( (w_eval1,w_eval2), axis=0 )

            eval_history = model.evaluate(x_eval,y_eval)
            eval_metric = eval_history

            eval_metrics += [eval_metric]
        train_metrics += [train_metric]
        
    return eval_metrics, train_metrics

# Expand small prediction to a whole image.
# _________________
# |       ____|
# |      | p  |
# |      |    |
# |---------(i,j)
# |
# as long as indexes are valid we can sample a patch p
# The patch_gap is the difference between input and outpu patch.
#     ___________
#    |  _______  |
#    | |       | |
#    | |       | |
#    | |_______| |
#    |___________|(i,j)
#
# If the outputs_patch_shape are smaller so they're allways valid.
#
def predict_full_image(model,x):
    h,w,_ = x.shape
    y = np.zeros([h,w], dtype=np.uint8)
    h_in,w_in,_ = model.inputs_shape[0]
    h_out,w_out,_ = model.outputs_shape[0]
    gap_h =(h_in-h_out)//2
    gap_w = (w_in-w_out)//2
    step_h = h_in + gap_h
    step_w = w_in + gap_w
    i,j = h_in,w_in
    while i<h:
        while j<w:
            # Extract patch from image
            patch_x = x[i-h_in:i,j-w_in:j]
            # Predict patch with model
            patch_y = np.argmax( model.predict(patch_x),axis=2)
            # Copy to y
            y[(i-h_in)+gap_h:i-gap_h,(j-w_in)+gap_w:j-gap_w] = patch_y

            j+=step_w
        i+=step_h
    return y

def to_image(x):
    mn,mx = np.min(x), np.max(x)
    return (x - mn)/float(mx-mn)

def from_categorical(batch_patch):
    return np.argmax(batch_patch,axis=3)

def show_batches(batches, batch_titles=None):
    fig = plt.figure()
    fig.set_size_inches(70.5, 70.5, forward=True)
    n_batches = len(batches)
    n_patches = batches[0].shape[0]
    for i in range(n_patches):
        for j in range(n_batches):
            a=fig.add_subplot(n_patches,n_batches,i*n_batches+j+1)
            image_dims = len(batches[j][i].shape)
            if image_dims==2:
                # RGB
                plt.imshow(batches[j][i]*255)
            else:
                # Greyscale
                plt.imshow(batches[j][i])
            if batch_titles is not None:
                a.set_title("{}_example_{}".format(batch_titles[j],i))
    plt.tight_layout()
    plt.show()
