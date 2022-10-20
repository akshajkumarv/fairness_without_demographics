import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from arguments import args


class WaterbirdsDataset(Dataset):
    
    def __init__(self, df, img_dir, transform=None):
    
        #df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        #self.csv_path = csv_path
        self.img_names = df['img_filename'].values
        self.y = df['y'].values
        self.attri = df['place'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)

        name = self.img_names[index]
        attri = self.attri[index]
        label = self.y[index]
        
        return img, name, attri, label

    def __len__(self):
        return self.y.shape[0]

BATCH_SIZE = args.batch_size

df = pd.read_csv('./waterbird_complete95_forest2water2/metadata.csv') 
train_split = df.loc[df['split'] == 0]
valid_split = df.loc[df['split'] == 1]
test_split = df.loc[df['split'] == 2]

custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_waterbkgd_mask = train_split['place'].apply(lambda x: x == 1)
train_landbkgd_mask = train_split['place'].apply(lambda x: x == 0)
train_waterbirds_mask = train_split['y'] == 1
train_landbirds_mask = train_split['y'] == 0

train_waterbkgd = train_split[train_waterbkgd_mask]
train_landbkgd = train_split[train_landbkgd_mask]
train_waterbirds_waterbkgd = train_split[train_waterbirds_mask & train_waterbkgd_mask]
train_landbirds_waterbkgd = train_split[train_waterbkgd_mask & train_landbirds_mask]
train_waterbirds_landbkgd = train_split[train_waterbirds_mask & train_landbkgd_mask]
train_landbirds_landbkgd = train_split[train_landbkgd_mask & train_landbirds_mask]
train_waterbirds = train_split[train_waterbirds_mask]
train_landbirds = train_split[train_landbirds_mask]

train_dataset = WaterbirdsDataset(df=train_split, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
train_waterbkgd_dataset = WaterbirdsDataset(df=train_waterbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
train_landbkgd_dataset = WaterbirdsDataset(df=train_landbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
train_waterbirds_waterbkgd_dataset = WaterbirdsDataset(df=train_waterbirds_waterbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
train_landbirds_waterbkgd_dataset = WaterbirdsDataset(df=train_landbirds_waterbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
train_waterbirds_landbkgd_dataset = WaterbirdsDataset(df=train_waterbirds_landbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
train_landbirds_landbkgd_dataset = WaterbirdsDataset(df=train_landbirds_landbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
train_waterbirds_dataset = WaterbirdsDataset(df=train_waterbirds, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
train_landbirds_dataset = WaterbirdsDataset(df=train_landbirds, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)


train_loader = DataLoader(dataset=train_dataset,
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  num_workers=4, 
                  pin_memory=True)      
train_waterbirds_waterbkgd_loader = DataLoader(dataset=train_waterbirds_waterbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=True,
                     num_workers=4)
train_waterbirds_landbkgd_loader = DataLoader(dataset=train_waterbirds_landbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=True,
                     num_workers=4)
train_landbirds_waterbkgd_loader = DataLoader(dataset=train_landbirds_waterbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=True,
                     num_workers=4)
train_landbirds_landbkgd_loader = DataLoader(dataset=train_landbirds_landbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=True,
                     num_workers=4)                     
train_waterbirds_loader = DataLoader(dataset=train_waterbirds_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=True,
                 num_workers=4, 
                 pin_memory=True)
train_landbirds_loader = DataLoader(dataset=train_landbirds_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=True,
                 num_workers=4, 
                 pin_memory=True)     

print("Shape of train dataset: ", train_split.shape)
print("Shape of train dataset with Waterbirds Hair", train_waterbirds.shape)
print("Shape of train dataset with Landbirds Hair", train_landbirds.shape)
print("Shape of train dataset with Waterbkgd and Waterbirds Hair: ", train_waterbirds_waterbkgd.shape)
print("Shape of train dataset with Landbkgd and Waterbirds Hair: ", train_waterbirds_landbkgd.shape)
print("Shape of train dataset with Waterbkgd and Landbirds Hair: ", train_landbirds_waterbkgd.shape)
print("Shape of train dataset with Landbkgd and Landbirds Hair: ", train_landbirds_landbkgd.shape)
print("Shape of train dataset with Waterbkgd: ", train_waterbkgd.shape)
print("Shape of train dataset with Landbkgd: ", train_landbkgd.shape)

# Unbiased training data
unbiased_train_split  = pd.DataFrame()   
unbiased_train_landbirds = train_landbirds.sample(n=1113, random_state=1)
unbiased_train_split = train_waterbirds.append(unbiased_train_landbirds, ignore_index=True)
unbiased_train_dataset = WaterbirdsDataset(df=unbiased_train_split, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
unbiased_train_landbirds_dataset = WaterbirdsDataset(df=unbiased_train_landbirds, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
unbiased_train_loader = DataLoader(dataset=unbiased_train_dataset,
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  num_workers=4, 
                  pin_memory=True)
unbiased_train_landbirds_loader = DataLoader(dataset=unbiased_train_landbirds_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=True,
                 num_workers=4, 
                 pin_memory=True)  
print("Shape of unbiased train dataset: ", unbiased_train_split.shape)
print("Shape of unbiased train landbirds dataset: ", unbiased_train_landbirds.shape)

valid_waterbkgd_mask = valid_split['place'].apply(lambda x: x == 1)
valid_landbkgd_mask = valid_split['place'].apply(lambda x: x == 0)
valid_waterbirds_mask = valid_split['y'] == 1
valid_landbirds_mask = valid_split['y'] == 0

valid_waterbkgd = valid_split[valid_waterbkgd_mask]
valid_landbkgd = valid_split[valid_landbkgd_mask]
valid_waterbirds_waterbkgd = valid_split[valid_waterbirds_mask & valid_waterbkgd_mask]
valid_landbirds_waterbkgd = valid_split[valid_waterbkgd_mask & valid_landbirds_mask]
valid_waterbirds_landbkgd = valid_split[valid_waterbirds_mask & valid_landbkgd_mask]
valid_landbirds_landbkgd = valid_split[valid_landbkgd_mask & valid_landbirds_mask]
valid_waterbirds = valid_split[valid_waterbirds_mask]
valid_landbirds = valid_split[valid_landbirds_mask]

valid_dataset = WaterbirdsDataset(df=valid_split, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
valid_waterbkgd_dataset = WaterbirdsDataset(df=valid_waterbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
valid_landbkgd_dataset = WaterbirdsDataset(df=valid_landbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
valid_waterbirds_waterbkgd_dataset = WaterbirdsDataset(df=valid_waterbirds_waterbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
valid_landbirds_waterbkgd_dataset = WaterbirdsDataset(df=valid_landbirds_waterbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
valid_waterbirds_landbkgd_dataset = WaterbirdsDataset(df=valid_waterbirds_landbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
valid_landbirds_landbkgd_dataset = WaterbirdsDataset(df=valid_landbirds_landbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
valid_waterbirds_dataset = WaterbirdsDataset(df=valid_waterbirds, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
valid_landbirds_dataset = WaterbirdsDataset(df=valid_landbirds, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)

valid_loader = DataLoader(dataset=valid_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=False,
                 num_workers=4, 
                 pin_memory=True)
valid_waterbirds_waterbkgd_loader = DataLoader(dataset=valid_waterbirds_waterbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)
valid_waterbirds_landbkgd_loader = DataLoader(dataset=valid_waterbirds_landbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)
valid_landbirds_waterbkgd_loader = DataLoader(dataset=valid_landbirds_waterbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)
valid_landbirds_landbkgd_loader = DataLoader(dataset=valid_landbirds_landbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)
valid_waterbirds_loader = DataLoader(dataset=valid_waterbirds_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=False,
                 num_workers=2, 
                 pin_memory=True)
valid_landbirds_loader = DataLoader(dataset=valid_landbirds_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=False,
                 num_workers=2, 
                 pin_memory=True)                 
valid_waterbkgd_loader = DataLoader(dataset=valid_waterbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)
valid_landbkgd_loader = DataLoader(dataset=valid_landbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)

print("Shape of valid dataset: ", valid_split.shape)
print("Shape of valid dataset with Waterbirds Hair", valid_waterbirds.shape)
print("Shape of valid dataset with Landbirds Hair", valid_landbirds.shape)
print("Shape of valid dataset with Waterbkgd and Waterbirds Hair: ", valid_waterbirds_waterbkgd.shape)
print("Shape of valid dataset with Landbkgd and Waterbirds Hair: ", valid_waterbirds_landbkgd.shape)
print("Shape of valid dataset with Waterbkgd and Landbirds Hair: ", valid_landbirds_waterbkgd.shape)
print("Shape of valid dataset with Landbkgd and Landbirds Hair: ", valid_landbirds_landbkgd.shape)
print("Shape of valid dataset with Waterbkgd: ", valid_waterbkgd.shape)
print("Shape of valid dataset with Landbkgd: ", valid_landbkgd.shape)

test_waterbkgd_mask = test_split['place'].apply(lambda x: x == 1)
test_landbkgd_mask = test_split['place'].apply(lambda x: x == 0)
test_waterbirds_mask = test_split['y'] == 1
test_landbirds_mask = test_split['y'] == 0

test_waterbkgd = test_split[test_waterbkgd_mask]
test_landbkgd = test_split[test_landbkgd_mask]
test_waterbirds_waterbkgd = test_split[test_waterbirds_mask & test_waterbkgd_mask]
test_landbirds_waterbkgd = test_split[test_waterbkgd_mask & test_landbirds_mask]
test_waterbirds_landbkgd = test_split[test_waterbirds_mask & test_landbkgd_mask]
test_landbirds_landbkgd = test_split[test_landbkgd_mask & test_landbirds_mask]
test_waterbirds = test_split[test_waterbirds_mask]
test_landbirds = test_split[test_landbirds_mask]

test_dataset = WaterbirdsDataset(df=test_split, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
test_waterbkgd_dataset = WaterbirdsDataset(df=test_waterbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
test_landbkgd_dataset = WaterbirdsDataset(df=test_landbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
test_waterbirds_waterbkgd_dataset = WaterbirdsDataset(df=test_waterbirds_waterbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
test_landbirds_waterbkgd_dataset = WaterbirdsDataset(df=test_landbirds_waterbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
test_waterbirds_landbkgd_dataset = WaterbirdsDataset(df=test_waterbirds_landbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
test_landbirds_landbkgd_dataset = WaterbirdsDataset(df=test_landbirds_landbkgd, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
test_waterbirds_dataset = WaterbirdsDataset(df=test_waterbirds, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)
test_landbirds_dataset = WaterbirdsDataset(df=test_landbirds, img_dir='./waterbird_complete95_forest2water2/', transform=custom_transform)

test_loader = DataLoader(dataset=test_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=False,
                 num_workers=4, 
                 pin_memory=True)
test_waterbirds_waterbkgd_loader = DataLoader(dataset=test_waterbirds_waterbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)
test_waterbirds_landbkgd_loader = DataLoader(dataset=test_waterbirds_landbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)
test_landbirds_waterbkgd_loader = DataLoader(dataset=test_landbirds_waterbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)
test_landbirds_landbkgd_loader = DataLoader(dataset=test_landbirds_landbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)
test_waterbirds_loader = DataLoader(dataset=test_waterbirds_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=False,
                 num_workers=4, 
                 pin_memory=True)
test_landbirds_loader = DataLoader(dataset=test_landbirds_dataset,
                 batch_size=BATCH_SIZE,
                 shuffle=False,
                 num_workers=4, 
                 pin_memory=True)
test_waterbkgd_loader = DataLoader(dataset=test_waterbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)
test_landbkgd_loader = DataLoader(dataset=test_landbkgd_dataset,
                     batch_size=BATCH_SIZE,
                     shuffle=False,
                     num_workers=4)

print("Shape of test dataset: ", test_split.shape)
print("Shape of test dataset with Waterbirds Hair", test_waterbirds.shape)
print("Shape of test dataset with Landbirds Hair", test_landbirds.shape)
print("Shape of test dataset with Waterbkgd and Waterbirds Hair: ", test_waterbirds_waterbkgd.shape)
print("Shape of test dataset with Landbkgd and Waterbirds Hair: ", test_waterbirds_landbkgd.shape)
print("Shape of test dataset with Waterbkgd and Landbirds Hair: ", test_landbirds_waterbkgd.shape)
print("Shape of test dataset with Landbkgd and Landbirds Hair: ", test_landbirds_landbkgd.shape)
print("Shape of test dataset with Waterbkgd: ", test_waterbkgd.shape)
print("Shape of test dataset with Landbkgd: ", test_landbkgd.shape)

