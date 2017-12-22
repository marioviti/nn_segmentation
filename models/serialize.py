from keras.models import model_from_json

def save_to(model,path_name):
    """
        -args:

        model: class model
        path_name : string : path/name
    """
    model_json = model.to_json()
    with open(path_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path_name+".h5")

def load_from(path_name):
    """
        -args:

        path_name : string : path/name
    """
    json_file = open(path_name+".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path_name+".h5")
    return loaded_model
