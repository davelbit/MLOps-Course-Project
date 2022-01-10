# -*- coding: utf-8 -*-
import argparse
import sys

import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import zipfile, io
import os

import requests
from tqdm import tqdm


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
