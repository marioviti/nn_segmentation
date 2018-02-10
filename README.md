# nn_segmentation

Image segmentation using neural networks.

## Notebooks

Notebooks provide an interface to train test either MIMO or UNET models.
there are 3 Notebooks:
* TrainModels
* TestModels
* VisualizeDataset

Each contains instruction of use.

## Implementation

The models Unet and MimoNet extend the class GeneralModel.
Hers's a snapshot of the main models to run:

[class_diagram]: https://ibb.co/d8Vnqn 

### Signature of most useful methods

* fit( x_train, y_train, batch_size=1, epochs=1, cropped=False, \*\*kwargs ): 
** x_train : batch_of_patch resized according to the input
** y_train : batch_of_patch resized according to the input/ouput
** cropped : if False y_train will be resized according to the output by the method
** kwargs : are passed to the wrapped [fit method](https://keras.io/getting-started/sequential-model-guide/#training)
** **output** : history object with losses and metrics 
* predict( x ): x : batch_of_patch resized according to the input
** **output** : y prediction resized according to the output
* save_model( name=None): name,string : for example ./trained_models/MIMO_500_c3
* load_model( name=None): name,string : for example ./trained_models/MIMO_500_c3


## Dependencies

* **requirements.txt** contains a list of python packages needed for executing the examples.
* check wheter you strictly needs some dependecies in **pipdeptree.txt**
* to install all dependecies
```pip install -r requirement.txt```

## HeadLess Script

### Optional arguments

To display help options: `python main.py -h`
* **-trn**,**--training**: specify if trainig session.
* **-ldmd**,**--load_model**: specify model path/name, the script will load the path/name.h5 and path/name.json files.

### Training Unet or MIMO on CD_Dataset

* In **main.py** set your parameters,  INPUT_PATCH_SIZE should not go over original height and width ( 2000x2000 ) These settings needs 1.5 GB of Vram minimum.:
```
INPUT_PATCH_SIZE = [350,350]
INPUT_CHANNELS = [3]
OUTPUT_CHANNELS = [2]
EPOCHS = 20
N_PATCH_BATCH = 10
```
* to execute simply run:
```
python main.py --training --load_model ./Unet
```

By default (for a training session) if no model is selected via the load_model option a new one is created with a random id of 10 alphanumerical digits and stored at the root of the project directory.


## Monitoring

### nvidia

* To monitor GPU usage **nvidia-smi**.
* To reseg GPU usage **nvidia-smi --gpu-reset -i 0**.
