# -*- coding: utf-8 -*-
import argparse
import sys

import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import requests, zipfile, io
import os


url="https://data.mendeley.com/public-files/datasets/jctsfj2sfn/files/148dd4e7-636b-404b-8a3c-6938158bc2c0/file_downloaded"

def download_extract(zip_file_url,PATH,filename="covid19-pneumonia-normal-chest-xraypa-dataset.zip",
foldername='COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset'):

    if PATH[-1] != '/':
        PATH.append('/')
    download=True
    extract=True
    if os.path.isfile(PATH+filename):
        download=False
    if os.path.isdir(PATH+foldername):
        extract=False

    if download:
        r = requests.get(zip_file_url)
        if extract:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(PATH)
    elif extract:
        z=zipfile.ZipFile(PATH+name)
        z.extractall(PATH)


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    parser = argparse.ArgumentParser(description='Data dwonloading and unzipping arguments')
    parser.add_argument("--url",type=str, default="https://data.mendeley.com/public-files/datasets/jctsfj2sfn/files/148dd4e7-636b-404b-8a3c-6938158bc2c0/file_downloaded",help="data URL to zip file")
    parser.add_argument("--PATH",type=str, default='../../data/raw/',help="where to save zip")
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
