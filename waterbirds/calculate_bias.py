import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

import torchvision.models as models

from arguments import args

from jtt_data import valid_waterbirds, valid_waterbirds_waterbkgd, valid_waterbirds_landbkgd, valid_landbirds, valid_landbirds_waterbkgd, valid_landbirds_landbkgd
from jtt_data import valid_loader, valid_waterbirds_loader, valid_waterbirds_waterbkgd_loader, valid_waterbirds_landbkgd_loader, valid_landbirds_loader, valid_landbirds_waterbkgd_loader, valid_landbirds_landbkgd_loader


def inverse_normalize(tensors):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)], std= [1/s for s in std])
    for i in range(tensors.shape[0]):
        tensors[i] = inv_normalize(tensors[i])
    
    return tensors

def calculate_mse_bias(stage_1_net, loader, device):
    correct_num_examples, incorrect_num_examples = 0.0, 0.0
    correct_features_sum, incorrect_features_sum = 0.0, 0.0
    
    with torch.no_grad():
        for i, (features, _, _, targets) in enumerate(loader):
            
            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).unsqueeze(1).float()
            logits = stage_1_net(features)
            probas = torch.sigmoid(logits)
            predicted_labels = probas > 0.5 
            
            correct = torch.where(predicted_labels == targets)[0]          
            incorrect = torch.where(predicted_labels != targets)[0]            
            
            features = inverse_normalize(features)
            features = features.mul(255).add_(0.5).clamp_(0, 255)#.permute(0, 2, 3, 1)

            correct_features = features[correct]
            correct_targets = targets[correct]
            incorrect_features = features[incorrect]
            incorrect_targets = targets[incorrect]
            
            correct_num_examples += correct_targets.size(0)
            incorrect_num_examples += incorrect_targets.size(0)
            correct_features_sum += torch.sum(correct_features, 0)
            incorrect_features_sum += torch.sum(incorrect_features, 0)

    correct_features_mean = correct_features_sum/correct_num_examples
    incorrect_features_mean = incorrect_features_sum/incorrect_num_examples
    
    mse = ((correct_features_mean - incorrect_features_mean)**2).mean()
    
    return correct_features_mean/255, incorrect_features_mean/255, mse.to('cpu').numpy()  
    
def difficult_examples(net, df, loader, epoch, device):
    with torch.no_grad():
        for i, (features, names, _, targets) in enumerate(loader):

            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).unsqueeze(1).float()
                        
            logits = net(features)
            probas = torch.sigmoid(logits)
            predicted_labels = probas > 0.5
             
            incorrect = torch.where(predicted_labels != targets)[0]            
            incorrect = list(incorrect.to('cpu').numpy())
            incorrect_names = [names[i] for i in incorrect]
            
            df.loc[df['img_filename'].isin(incorrect_names), 'mistakes_epoch_%s'%(epoch)] = 1
        
    return df 

# Hyperparameters
DEVICE = 'cuda:0'
BASE_LR = args.lr
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MOMENTUM = args.momentum
WEIGHT_DECAY = args.weight_decay
RUN = args.run

PATH = 'saved_models/run_%s/bs_%s_epochs_%s_lr_%s_wd_%s_class_balanced/'%(RUN, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY)
if not os.path.exists(PATH+'statistics/'):
    os.makedirs(PATH+'statistics/')    

mses = []        
waterbirds_mses = []
landbirds_mses = []
waterbirds_waterbkgd_mses = []
waterbirds_landbkgd_mses = []
landbirds_waterbkgd_mses = []
landbirds_landbkgd_mses = []

df_waterbirds = valid_waterbirds.copy()
df_waterbirds_waterbkgd = valid_waterbirds_waterbkgd.copy()
df_waterbirds_landbkgd = valid_waterbirds_landbkgd.copy()
df_landbirds = valid_landbirds.copy()
df_landbirds_waterbkgd = valid_landbirds_waterbkgd.copy()
df_landbirds_landbkgd = valid_landbirds_landbkgd.copy()


for STAGE_1_MODEL in range(1, 301):
    if not os.path.exists(PATH + 'statistics/epoch_%s/'%(STAGE_1_MODEL)):
        os.makedirs(PATH + 'statistics/epoch_%s/'%(STAGE_1_MODEL))    

    stage_1_model = models.resnet50(pretrained=True)
    d = stage_1_model.fc.in_features
    stage_1_model.fc = nn.Linear(d, 1)
    if torch.cuda.device_count() > 1:
        stage_1_model = nn.DataParallel(stage_1_model, device_ids=[0])
    stage_1_model.to(DEVICE)    
    load_path = PATH + 'epoch_%s/epoch_%s.pt'%(STAGE_1_MODEL, STAGE_1_MODEL)
    print('Loading Model Checkpoint: ', load_path) 
    checkpoint = torch.load(load_path) 
    stage_1_model.load_state_dict(checkpoint['model_state_dict'])
    stage_1_model.eval()

    _, _, mse = calculate_mse_bias(stage_1_model, valid_loader, DEVICE)
    mses.append(mse)
    np.savetxt(PATH + "statistics/mses.txt", mses)    
        
    correct_features_mean, incorrect_features_mean, waterbirds_mse = calculate_mse_bias(stage_1_model, valid_waterbirds_loader, DEVICE)
    waterbirds_mses.append(waterbirds_mse)
    np.savetxt(PATH + "statistics/waterbirds_mses.txt", waterbirds_mses)    
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/waterbirds_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/waterbirds_incorrect_features_mean.png'%(STAGE_1_MODEL))
    
    correct_features_mean, incorrect_features_mean, landbirds_mse = calculate_mse_bias(stage_1_model, valid_landbirds_loader, DEVICE)    
    landbirds_mses.append(landbirds_mse)
    np.savetxt(PATH + "statistics/landbirds_mses.txt", landbirds_mses)
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/landbirds_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/landbirds_incorrect_features_mean.png'%(STAGE_1_MODEL))

    correct_features_mean, incorrect_features_mean, waterbirds_waterbkgd_mse = calculate_mse_bias(stage_1_model, valid_waterbirds_waterbkgd_loader, DEVICE)
    waterbirds_waterbkgd_mses.append(waterbirds_waterbkgd_mse)
    np.savetxt(PATH + "statistics/waterbirds_waterbkgd_mses.txt", waterbirds_waterbkgd_mses)    
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/waterbirds_waterbkgd_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/waterbirds_waterbkgd_incorrect_features_mean.png'%(STAGE_1_MODEL))

    correct_features_mean, incorrect_features_mean, waterbirds_landbkgd_mse = calculate_mse_bias(stage_1_model, valid_waterbirds_landbkgd_loader, DEVICE)
    waterbirds_landbkgd_mses.append(waterbirds_landbkgd_mse)
    np.savetxt(PATH + "statistics/waterbirds_landbkgd_mses.txt", waterbirds_landbkgd_mses)    
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/waterbirds_landbkgd_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/waterbirds_landbkgd_incorrect_features_mean.png'%(STAGE_1_MODEL))

    correct_features_mean, incorrect_features_mean, landbirds_waterbkgd_mse = calculate_mse_bias(stage_1_model, valid_landbirds_waterbkgd_loader, DEVICE)    
    landbirds_waterbkgd_mses.append(landbirds_waterbkgd_mse)
    np.savetxt(PATH + "statistics/landbirds_waterbkgd_mses.txt", landbirds_waterbkgd_mses)
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/landbirds_waterbkgd_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/landbirds_waterbkgd_incorrect_features_mean.png'%(STAGE_1_MODEL))

    correct_features_mean, incorrect_features_mean, landbirds_landbkgd_mse = calculate_mse_bias(stage_1_model, valid_landbirds_landbkgd_loader, DEVICE)    
    landbirds_landbkgd_mses.append(landbirds_landbkgd_mse)
    np.savetxt(PATH + "statistics/landbirds_landbkgd_mses.txt", landbirds_landbkgd_mses)
    save_image(correct_features_mean, PATH + 'statistics/epoch_%s/landbirds_landbkgd_correct_features_mean.png'%(STAGE_1_MODEL))
    save_image(incorrect_features_mean, PATH + 'statistics/epoch_%s/landbirds_landbkgd_incorrect_features_mean.png'%(STAGE_1_MODEL))

    df_waterbirds['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_waterbirds = difficult_examples(stage_1_model, df_waterbirds, valid_waterbirds_loader, STAGE_1_MODEL, DEVICE)
    df_waterbirds_waterbkgd['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_waterbirds_waterbkgd = difficult_examples(stage_1_model, df_waterbirds_waterbkgd, valid_waterbirds_waterbkgd_loader, STAGE_1_MODEL, DEVICE)
    df_waterbirds_landbkgd['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_waterbirds_landbkgd = difficult_examples(stage_1_model, df_waterbirds_landbkgd, valid_waterbirds_landbkgd_loader, STAGE_1_MODEL, DEVICE)

    df_landbirds['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_landbirds = difficult_examples(stage_1_model, df_landbirds, valid_landbirds_loader, STAGE_1_MODEL, DEVICE)
    df_landbirds_waterbkgd['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_landbirds_waterbkgd = difficult_examples(stage_1_model, df_landbirds_waterbkgd, valid_landbirds_waterbkgd_loader, STAGE_1_MODEL, DEVICE)
    df_landbirds_landbkgd['mistakes_epoch_%s'%(STAGE_1_MODEL)] = 0
    df_landbirds_landbkgd = difficult_examples(stage_1_model, df_landbirds_landbkgd, valid_landbirds_landbkgd_loader, STAGE_1_MODEL, DEVICE)

df_waterbirds['total_mistakes'] = df_waterbirds.iloc[:,6:].sum(axis=1)
df_waterbirds.to_csv(PATH + "statistics/df_waterbirds.csv", index=False)

df_waterbirds_waterbkgd['total_mistakes'] = df_waterbirds_waterbkgd.iloc[:,6:].sum(axis=1)
df_waterbirds_waterbkgd.to_csv(PATH + "statistics/df_waterbirds_waterbkgd.csv", index=False)

df_waterbirds_landbkgd['total_mistakes'] = df_waterbirds_landbkgd.iloc[:,6:].sum(axis=1)
df_waterbirds_landbkgd.to_csv(PATH + "statistics/df_waterbirds_landbkgd.csv", index=False)

df_landbirds['total_mistakes'] = df_landbirds.iloc[:,6:].sum(axis=1)
df_landbirds.to_csv(PATH + "statistics/df_landbirds.csv", index=False)

df_landbirds_waterbkgd['total_mistakes'] = df_landbirds_waterbkgd.iloc[:,6:].sum(axis=1)
df_landbirds_waterbkgd.to_csv(PATH + "statistics/df_landbirds_waterbkgd.csv", index=False)

df_landbirds_landbkgd['total_mistakes'] = df_landbirds_landbkgd.iloc[:,6:].sum(axis=1)
df_landbirds_landbkgd.to_csv(PATH + "statistics/df_landbirds_landbkgd.csv", index=False)

