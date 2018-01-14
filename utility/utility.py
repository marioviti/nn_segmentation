from matplotlib import pyplot as plt
import numpy as np

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
    h,w = x.shape
    y = np.zeros([h,w], dtype=np.uint8)
    h_in,w_in,_ = model.input_shape
    h_out,w_out,_ = model.outputs_shape
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
            patch_y = np.argmax( model.predict(x),axis=2)
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
