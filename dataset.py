import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import pandas as pd
from PIL import Image

class Animal10(Dataset):
    def __init__(self,is_train,image_path="./images/raw-img/", transform = None):
        self.img_folder = image_path
        self.class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
        self.class_idx = {cls_name:i for i,cls_name in enumerate(self.class_names)}
        self.transform = transform
        if is_train:
            self.img_df = pd.read_csv('./images/train.csv')
        else:
            self.img_df = pd.read_csv('./images/test.csv')
    def __len__(self):
        return len(self.img_df)
    
    def __getitem__(self, idx):
        img_serial = self.img_df.iloc[idx]
        
        img_path = self.img_folder + img_serial['path']
        img_data = Image.open(img_path).convert('RGB')
        if self.transform:
            img_data = self.transform(img_data)

        # label_data = torch.zeros(10)
        # label_data[self.class_idx[img_serial['class']]] = 1
        label_data = self.class_idx[img_serial['class']]
        # print(img_data.shape, label_data)
        return img_data,label_data
