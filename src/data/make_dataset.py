# -*- coding: utf-8 -*-
import argparse
import sys

import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
import torchvision 
import zipfile, io
import os
import numpy as np
import requests
from tqdm import tqdm
import numpy as np

import kornia as K
import cv2
from sklearn.model_selection import train_test_split

url="https://data.mendeley.com/public-files/datasets/jctsfj2sfn/files/148dd4e7-636b-404b-8a3c-6938158bc2c0/file_downloaded"

def download_extract(zip_file_url : str,PATH,filename : str ="covid19-pneumonia-normal-chest-xraypa-dataset.zip",
foldername : str ='COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset',
chunk_size : int =1024):
    """
    Script to download dataset zip into raw folder
    zip_file_url : url to download file from
    PATH : path to download to
    filename : wanted filename for zip
    foldername : unzipped foldername
    inspired by: https://gist.github.com/nikhilkumarsingh/d29c1fdec0f4e266e53137d96b52e289
    """
    print(os.getcwd())
    print(os.path.isdir(PATH))
    print(os.listdir(PATH))
    if PATH[-1] != '/':
        PATH.append('/')
    
    print()
    print(PATH+filename)
    print()

    download=True
    extract=True
    if os.path.isfile(PATH+filename):
        download=False
    if os.path.isdir(PATH+foldername):
        extract=False

    if download:
        r = requests.get(zip_file_url, stream = True)
        total_size = int(r.headers['content-length'])
        with open(PATH+filename, 'wb') as f:
            for data in tqdm(iterable = r.iter_content(chunk_size = chunk_size), total = total_size/chunk_size, unit = 'KB'):
                f.write(data)
        print("Download complete!")

    if extract:
        z=zipfile.ZipFile(PATH+filename)
        z.extractall(PATH)

def preprocess(path : str):
    classes=[ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
    classes=np.sort(classes)
    class_map={name:c for c,name in enumerate(classes)}
    img_paths=[]
    labels=[]
    for class_name in class_map:
        path_=path+'/'+class_name
        files_names=[ name for name in os.listdir(path_) if not os.path.isdir(os.path.join(path_, name)) ]
        for file_name in files_names:
            img_paths.append(path_+'/'+file_name)
            labels.append(class_map[class_name])
    
    resize=torchvision.transforms.Resize((512,512))
    gray=torchvision.transforms.Grayscale(num_output_channels=1)
    

    all_images_gray512=torch.empty([len(img_paths),512,512])

    # path='../../data/raw/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset'
    #path='/data/raw/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset'
    for c,i in enumerate(tqdm(img_paths)):
        img_bgr: np.array = cv2.imread(i)
        if (img_bgr[0]==img_bgr[1]).all() and (img_bgr[1]==img_bgr[2]).all():
            img_gray=img_bgr[:,:,0]
        else:
            img_gray: np.array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
        x_gray: torch.tensor = K.image_to_tensor(img_gray).view(1,*img_gray.shape)

        img_gray512: torch.tensor=resize(x_gray)
        all_images_gray512[c,:,:]=img_gray512
    #     batch=torch.cat((batch, img_gray512), 0)
    #     if c%100==0 &c>1:
    #         all_images_gray512=torch.cat((all_images_gray512, batch), 0)
    #         if c!=len(all_images_gray512):
    #             print(c+1,len(all_images_gray512))
    #             raise BrokenPipeError("image was missing")
    #         del batch
    #         batch=torch.Tensor()
    # all_images_gray512=torch.cat((all_images_gray512, batch), 0)

     # split train set in test and validation set
    validation_split = .3
    seed = 42
    train_indices, validation_indices, _, _ = train_test_split(
        range(len(all_images_gray512)),
        labels,
        stratify=labels,
        test_size=validation_split,
        random_state=seed
    )
    train_images=all_images_gray512[train_indices]
    test_images=all_images_gray512[validation_indices]
    train_labels=np.array(labels)[train_indices]
    test_labels=np.array(labels)[validation_indices]

    output_filepath='data/preprocessed/covid_not_norm/'
    if not os.path.isdir(output_filepath):
        os.makedirs(output_filepath)
    torch.save(train_images.float(),output_filepath+'train_images.pt')
    torch.save(torch.from_numpy(train_labels),output_filepath+'train_labels.pt')

    torch.save(test_images.float(),output_filepath+'test_images.pt')
    torch.save(torch.from_numpy(test_labels),output_filepath+'test_labels.pt')


    

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    parser = argparse.ArgumentParser(description='Data dwonloading and unzipping arguments')
    parser.add_argument("--url",type=str, default="https://data.mendeley.com/public-files/datasets/jctsfj2sfn/files/148dd4e7-636b-404b-8a3c-6938158bc2c0/file_downloaded",help="data URL to zip file")
    parser.add_argument("--PATH",type=str, default='data/raw/',help="where to save zip")
    parser.add_argument("--NAME", type=str, default="covid19-pneumonia-normal-chest-xraypa-dataset.zip", help="name of file to be extracted")
    parser.add_argument("--exdir", type=str, default='COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset', help="name of dir to be extracted")

    args = parser.parse_args(sys.argv[2:])
    zip_file_url=args.url
    PATH=args.PATH
    filename=args.NAME
    foldername=args.exdir
    download_extract(zip_file_url,PATH,filename,foldername)
    path='data/raw/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset'
    preprocess(path)

    logger = logging.getLogger(__name__)

    
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
