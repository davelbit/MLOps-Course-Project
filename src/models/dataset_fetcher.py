import torch
from torch.jit import Error
from torch.utils.data import Dataset
import numpy as np

import torchvision

import os


class Dataset_fetcher(Dataset):
    def __init__(self,path,transform=None):

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
        
        self.labels=np.array(labels).copy()
        self.img_paths=np.array(img_paths).copy()


        self.transform=transform
        self.resizeandgray=torchvision.transforms.Compose(
            [torchvision.transforms.Resize((512,512)),
            torchvision.transforms.Grayscale(num_output_channels=1)])
        self.resize=torchvision.transforms.Resize((512,512))
        self.gray=torchvision.transforms.Grayscale(num_output_channels=1)

        
    def __getitem__(self, idx):
        try:
            x_rgb: torch.tensor = torchvision.io.read_image(self.img_paths[idx])  # CxHxW / torch.uint8
        except:
            print(self.img_paths[idx])
            print("image not supported")
            return False, False
            ##some jpgs are actually pngs
            ##but some pngs are not 8 bit so another error...
            # png_name=self.img_paths[idx][:self.img_paths[idx].rfind('.')]+'.png'
            # os.rename(self.img_paths[idx],png_name)
            # print(png_name,os.path.isfile(png_name))
            # x_rgb: torch.tensor = torchvision.io.read_image(png_name)
        # x_rgb = x_rgb.unsqueeze(0)  # BxCxHxW
        x_rgb=self.resize(x_rgb)
        # print(x_rgb.shape)
        try:
            if x_rgb.shape[0]!=1:
                if (x_rgb[1]==x_rgb[2]).all():
                    x_rgb=x_rgb[0].view(1,*x_rgb[0].shape)
                else:
                    x_rgb=self.gray(x_rgb[:3])
        except:
            print(self.img_paths[idx])
            print(x_rgb.shape)

        y = self.labels[idx]

        if self.transform:
            x_rgb = self.transform(x_rgb)

        return x_rgb.float(),y

    def __len__(self):
        return (len(self.img_paths))


# DS=Dataset_fetcher('../../data/raw/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset')
# loader = torch.utils.data.DataLoader(DS, shuffle=False, num_workers=0, batch_size=3)

# dataiter = iter(loader)
# images, labels = dataiter.next()

# print( images.shape,labels.shape)
