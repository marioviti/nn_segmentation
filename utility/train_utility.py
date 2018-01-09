import numpy as np

def train(model, datamanager, epochs=10, n_batch=5):
    h,w,_ = model.input_shape
    print(h,w)
    print(n_batch)
    metrics = np.zeros(epochs)
    losses = np.zeros(epochs)
    for i in range(epochs):
        print("================epoch {}=================".format(i))
        x_train,y_train = datamanager.sample_X_Y_patch_batch([h,w],
                                        n_batch=n_batch)
        model.fit(x_train,y_train)
        model.evaulate(x_train,y_train)
        loss, metric = model.score
        metrics[i] = metric
        losses[i] = loss
    return metrics,losses

def train_and_save(model, datamanager, name, epochs=10, n_batch=5):
    h,w,_ = model.input_shape
    print(h,w)
    print(n_batch)
    metrics = np.zeros(epochs)
    losses = np.zeros(epochs)
    for i in range(epochs):
        print("================epoch {}/{}=================".format(i,epochs))
        if i%10 == 0:
            print('saving model: {}'.format(name))
            model.save_model(name)
        x_train,y_train = datamanager.sample_X_Y_patch_batch([h,w],
                                        n_batch=n_batch)
        model.fit(x_train,y_train)
        model.evaulate(x_train,y_train)
        loss, metric = model.score
        metrics[i] = metric
        losses[i] = loss
    return metrics,losses
