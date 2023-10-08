import cv2
import torch
from torch.utils.data import Dataset

from dataset.custom_transforms import *


class DeepFakesDataset(Dataset):
    def __init__(self, images, labels, transform = None):
        self.x = images
        self.y = labels
        self.transform = eval(transform) if isinstance(transform,str) else transform
    
     
    def __getitem__(self, index):
        image = cv2.imread(self.x[index])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        if(self.transform is not None):
            image = self.transform(image=image)['image']
        
        return torch.tensor(image).float(), self.y[index]


    def __len__(self):
        return len(self.x)

 