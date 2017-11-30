# nn_segmentation


Project focused on image segmentation using neural networks.

## Dependencies

* **requirements.txt** contains a list of python packages needed for executing the examples.
* check wheter you strictly needs some dependecies in **pipdeptree.txt**
* to install all dependecies
```pip install -r requirement.txt```


## Usage

### Training Unet on CD_Dataset

* In **main.py** set your parameters,  INPUT_PATCH_SIZE should not go over original height and width ( 2000x2000 ) These settings needs 1.5 GB of Vram minimum.:
```
INPUT_PATCH_SIZE = [350,350]
INPUT_CHANNELS = [3]
EPOCHS = 10
N_PATCH_BATCH = 5
```
* to execute simply run:
```
python main.py
```

## Monitoring

### nvidia

* To monitor GPU usage **nvidia-smi**.
* To reseg GPU usage **nvidia-smi --gpu-reset -i 0**.
