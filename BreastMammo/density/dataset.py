import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import torch
import torchvision

def IU(root='/dataset/Mammogram/IU_PNG', view = 1):
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

def IU_birads(root='/dataset/Mammogram/IU_PNG'):
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
            data['label'].append(int(df['BIRADS'][i]))
            data['lateral'].append(lateral)
    return data

def IU_density(root='/dataset/Mammogram/IU_PNG'):
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
    data = IU()
    transforms = v2.Compose([
        v2.ToImage(), 
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=3),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    ds = ImageDataset(data, transform = transforms)
    