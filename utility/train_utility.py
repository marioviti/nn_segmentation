import numpy as np

def train(model, datamanager, epochs=10, n_batch=5):
    h,w,_ = model.input_shape
    print(h,w)
    print(n_batch)
    metrics = np.zeros(epochs)
    losses = np.zeros(epochs)
    for i in range(epochs):
        print("================epoch {}=================".format(i))
        x_train,y_train,w_train = datamanager.sample_X_Y_W_patch_batch([h,w],
                                        n_batch=n_batch)
        model.fit(x_train,y_train)
        model.evaulate(x_train,y_train)
        loss, metric = model.score
        metrics[i] = metric
        losses[i] = loss
    return metrics,losses

def train_and_save(model, dataset, name, epochs=10, n_batch=5):
    h,w,_ = model.inputs_shape[0]
    train_metrics = np.zeros(epochs)
    train_losses = np.zeros(epochs)
    eval_metrics = np.zeros(epochs)
    eval_losses = np.zeros(epochs)
    for i in range(epochs):
        print("================epoch {}/{}=================".format(i+1,epochs))
        if (i+1)%10 == 0:
            print('saving model: {}'.format(name))
            model.save_model(name)
        x_train,y_train,w_train = dataset.sample_X_Y_W_patch_batch([h,w],
                                        n_batch=n_batch)
        train_history = model.fit(x_train,y_train)
        train_history =train_history.history.values()

        x_eval,y_eval,w_eval = dataset.sample_X_Y_W_patch_batch([h,w],
                                                            n_batch=n_batch,
                                                            train=False)

        eval_history = model.evaluate(x_eval,y_eval)

    return train_losses,eval_losses,train_metrics,eval_metrics
