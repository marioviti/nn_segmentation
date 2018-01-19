import wget
import zipfile
import os.path as pth

data_url = data_url = 'https://www.dropbox.com/s/pxz42kgrt7oltix/CD_Dataset_01.zip?dl=1'

data_name = 'CD_Dataset_01.zip'
downloaded_data_path = 'CD_Dataset_01.zip'

def download(output='./'):
    downloaded_data_path = data_name
    downloaded_data_path = pth.join(output,downloaded_data_path)
    wget.download(data_url,out=output)

def unzip(filename=downloaded_data_path):
    a = zipfile.ZipFile(filename)
    a.extractall()

def dwuzp():
    download()
    unzip()
