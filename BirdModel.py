import os
import cv2
import torchvision

import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


from .config import DEVICE, ResNet


class BirdDataset(Dataset):
    def __init__(self, dataset_dir: str, img_dir: str, transform=None):
        self.df = pd.read_csv(dataset_dir)
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, i: int):
        # The Dataframe holds the names of the image associated to each entry
        self.img_names = self.df["filename"]
        name = self.img_names[i]
        image = Image.open(os.path.join(os.path.join(os.getcwd(), self.img_dir), name))
        image = np.asarray(image).astype(np.uint8)
        
        # Some Images are in gray or not well resized in the given dataset
        if image.shape[0] > image.shape[1]:
            image = np.rot90(image).copy()
        if len(image.shape) > 2 and image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        image = np.resize(image, (300, 500, 3))
        image = np.transpose(image, (2, 0, 1))
        
        if self.transform:
            image = self.transform(image)
    
        return [ np.asarray(image).astype(np.uint8), self.df["cls"].iloc[i]]

    
    def __len__(self):
        return self.df.shape[0]



class BirdModel(nn.Module):

    def __init__(self, num_classes: int, pretrained=True):
        super().__init__()
        
        # We want to use ResNet to help differenciating birds, 
        # so we link the output of ResNet to a simple new classifyer
        self.network = torchvision.models.resnet34(weights='IMAGENET1K_V1').to(DEVICE)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes).to(DEVICE)

    def forward(self, net: ResNet):
        return self.network( net )