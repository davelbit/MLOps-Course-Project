import json
import numpy as np
import torch
from torch import nn
import omegaconf
from omegaconf import OmegaConf
import requests
import os
import matplotlib.pyplot as plt


config = OmegaConf.load("config/config.yaml")
BASE_DIR = os.getcwd()
VALID_PATHS = {
        "images": BASE_DIR + config.VALID_PATHS.images,
        "labels": BASE_DIR + config.VALID_PATHS.labels,
    }

randomint=np.random.randint(20)
img=torch.load(VALID_PATHS["images"])[randomint]
labl=torch.load(VALID_PATHS["labels"])[randomint].numpy().tolist()

# img=np.round(np.random.normal(0,1,[512,512]),1)

img=img.view(512,512).numpy()

plt.figure()
plt.imshow(img,cmap='gray')
plt.show()
x={'input_data':img.tolist(),'model-id':"models/D19012022T170140best_model.pth"}

url='https://europe-west1-charged-city-337910.cloudfunctions.net/COVID-1'
url='https://europe-west1-charged-city-337910.cloudfunctions.net/COVID-predict'
# url="http://localhost:8081/"
diagnosis={0:'Covid',1:'Normal',2:'Pneumonia'}
print("Truth: ",diagnosis[labl] )


r=requests.post(url,json=x)
print( r.text)

"functions-framework --target=predict_covid --port=8081"