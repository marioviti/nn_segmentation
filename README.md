# nn_segmentation


Project focused on image segmentation using neural networks.

## Dependencies

* **requirements.txt** contains a list of python packages needed for executing the examples.
* check wheter you strictly needs some dependecies in **pipdeptree.txt**
* to install all dependecies
```pip install -r requirement.txt```


## Usage

### Optional arguments

* **-trn**,**--training**: specify if trainig session.
* **-tst**,**--testing**: specify if testing session.
* **-ldmd**,**--load_model**: specify model path/name, the script will load the path/name.h5 and path/name.json files.

### Training Unet on CD_Dataset

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

### testing

If a desktop environment is available, visualize the results of the trained network Unet
* to execute simply run:
```
python main.py --testing --load_model ./Unet
```

## Monitoring

### nvidia

* To monitor GPU usage **nvidia-smi**.
* To reseg GPU usage **nvidia-smi --gpu-reset -i 0**.
