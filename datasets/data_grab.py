import wget
import zipfile
import os.path as pth

data_url = data_url = 'https://www.dropbox.com/s/pxz42kgrt7oltix/CD_Dataset_01.zip?dl=1'

data_name = 'CD_Dataset_01.zip'

def download(output='./'):
    wget.download(data_url,out=output)    

def unzip(filename=data_name):
    a = zipfile.ZipFile(filename)
    a.extractall()

def dwuzp(path='./'):
    download(output=path)
    unzip(filename=pth.join(path,data_name))
