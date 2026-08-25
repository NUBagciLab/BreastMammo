import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torch
import torchvision
import numpy as np

def BreastMammo(root='/dataset/Mammogram/BreastMammo_PNG', view = 1):
    if view not in [1, 2]:
        raise ValueError(f"View must be 1 or 2 but got {view}!")
    data = {'image': [], 'label': [], 'lateral': []}
    for label in ['Benign', 'Malign']:
        if label == 'Benign':
            label_value = 0.0
        else:
            label_value = 1.0
        df = pd.read_excel(os.path.join(root, label+'_Cases.xlsx'), dtype=str)
        for i in range(len(df)):
            if df['RIGHT_OR_LEFT'][i] == '2':
                lateral = 'L'
            elif df['RIGHT_OR_LEFT'][i] == '1':
                lateral = 'R'
            else:
                raise ValueError(f"RIGH_OR_LEFT must be 1 or 2 but got {df['RIGHT_OR_LEFT'][i]} at {label} ID={df['ID'][i]}!")
            if view == 1:
                for v in ['CC', 'MLO']:
                    data['image'].append(os.path.join(root, label, df['ID'][i]+lateral+v+'.png'))
                    data['label'].append(label_value)
                    data['lateral'].append(lateral)
            elif view == 2:
                data['image'].append([os.path.join(root, label, df['ID'][i]+lateral+'CC.png'), os.path.join(root, label, df['ID'][i]+lateral+'MLO.png')])
                data['label'].append(label_value)
                data['lateral'].append(lateral)
            else:
                raise ValueError(f"View must be 1 or 2 but got {view}!")
    return data

def BreastMammo_density(root='/dataset/Mammogram/BreastMammo_PNG'):
    data = {'image': [], 'label': [], 'lateral': []}
    for label in ['Benign', 'Malign']:
        df = pd.read_excel(os.path.join(root, label+'_Cases.xlsx'), dtype=str)
        for i in range(len(df)):
            if df['RIGHT_OR_LEFT'][i] == '2':
                lateral = 'L'
            elif df['RIGHT_OR_LEFT'][i] == '1':
                lateral = 'R'
            else:
                raise ValueError(f"RIGH_OR_LEFT must be 1 or 2 but got {df['RIGHT_OR_LEFT'][i]} at {label} ID={df['ID'][i]} {i}!")
            data['image'].append([os.path.join(root, label, df['ID'][i]+lateral+'CC.png'), os.path.join(root, label, df['ID'][i]+lateral+'MLO.png')])
            data['label'].append(int(df['BREAST COMPOSITION'][i])-1)
            data['lateral'].append(lateral)
    return data

def DenseMammo_density(root='/dataset/Mammogram/DenseMammo_PNG'):
    data = {'image': [], 'label': [], 'lateral': []}
    for label in ['Benign', 'Malign']:
        df = pd.read_excel(os.path.join(root, label+'_Cases.xlsx'), dtype=str)
        for i in range(len(df)):
            for lateral in ['L', 'R']:
                data['image'].append([os.path.join(root, label, df['ID'][i]+lateral+'CC.png'), os.path.join(root, label, df['ID'][i]+lateral+'MLO.png')])
                data['label'].append(int(df['BREAST COMPOSITION'][i])-1)
                data['lateral'].append(lateral)
    return data

def TNMammo(root='/dataset/Mammogram/TNMammo_Clean'):
    data = {'image': [], 'label': [], 'lateral': []}
    df = pd.read_csv(os.path.join(root, 'TNMammo_labels.csv'))
    for i in range(len(df)):
        data['image'].append([os.path.join(root, 'images', str(df['ID'][i])+'left_cc.jpg'), os.path.join(root, 'images', str(df['ID'][i])+'left_mlo.jpg')])
        data['label'].append(ord(df['Labels'][i])-ord('A'))
        data['lateral'].append('L')
        data['image'].append([os.path.join(root, 'images', str(df['ID'][i])+'right_cc.jpg'), os.path.join(root, 'images', str(df['ID'][i])+'right_mlo.jpg')])
        data['label'].append(ord(df['Labels'][i])-ord('A'))
        data['lateral'].append('R')
    return data

def LUMINA_Density(root='/dataset/Mammogram/LUMINA_PNG'):
    data = {'image': [], 'label': [], 'lateral': []}
    for label in ['Benign', 'Malign']:
        df = pd.read_excel(os.path.join(root, label+'_Cases.xlsx'), dtype=str)
        for i in range(0, len(df), 2):     
            if df['RIGHT_OR_LEFT'][i] == 'BILATERAL':         
                for file in ['L', 'R']:
                    data['image'].append([os.path.join(root, label, df['ID'][i]+file+'_CC.png'),
                                    os.path.join(root, label, df['ID'][i]+file+'_MLO.png')])                          
                    data['label'].append(np.int64(df['BREAST COMPOSITION'][i])-1)       
                    data['lateral'].append(file)
                    
            elif df['RIGHT_OR_LEFT'][i] == 'LEFT':
                data['image'].append([os.path.join(root, label, df['ID'][i]+'L_CC.png'),
                                os.path.join(root, label, df['ID'][i]+'L_MLO.png')])                          
                data['label'].append(np.int64(df['BREAST COMPOSITION'][i])-1)       
                data['lateral'].append('L')
                         
            elif df['RIGHT_OR_LEFT'][i] == 'RIGHT':
                data['image'].append([os.path.join(root, label, df['ID'][i]+'R_CC.png'),
                                os.path.join(root, label, df['ID'][i]+'R_MLO.png')])                          
                data['label'].append(np.int64(df['BREAST COMPOSITION'][i])-1)       
                data['lateral'].append('R')
    return data

class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        image = Image.open(self.data['image'][idx])
        if self.data['lateral'][idx] == 'R':
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.transform:
            image = self.transform(image)
        label = self.data['label'][idx]
        return image, label
    
class MultiImagesDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        image = [Image.open(i) for i in self.data['image'][idx]]
        if self.data['lateral'][idx] == 'R':
            image = [i.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for i in image]
        image = [torchvision.tv_tensors.Image(i) for i in image]
        if self.transform:
            image = self.transform(image)
        label = self.data['label'][idx]
        return image, label
    
def get_fold(data, n_splits = 5, fold = 0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    skf.get_n_splits(data['image'], data['label'])
    for i, (train_index, test_index) in enumerate(skf.split(data['image'], data['label'])):
        if i == fold:
            train_data = {
                key: [data[key][j] for j in train_index] 
                for key in data
            }
            test_data = {
                key: [data[key][j] for j in test_index] 
                for key in data
            }
    return train_data, test_data

if __name__ == "__main__":
    data = BreastMammo_density()
    transforms = v2.Compose([
        v2.ToImage(), 
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=3),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    ds = ImageDataset(data, transform = transforms)
    