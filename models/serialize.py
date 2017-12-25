from keras.models import model_from_json

def save_to(model,name):
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+".h5")
    print("Saved model to disk")

def load_from(name):
    json_file = open(name+".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name+".h5")
    return loaded_model
    print("Loaded model from disk")
