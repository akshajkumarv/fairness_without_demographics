import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time

import numpy as np
import pandas as pd

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
       
from jtt_data import valid_loader, valid_waterbirds_loader, valid_waterbirds_waterbkgd_loader, valid_waterbirds_landbkgd_loader, valid_waterbkgd_loader, valid_landbkgd_loader, valid_landbirds_loader, valid_landbirds_waterbkgd_loader, valid_landbirds_landbkgd_loader, test_loader, test_waterbirds_waterbkgd_loader, test_waterbirds_landbkgd_loader, test_landbirds_waterbkgd_loader, test_landbirds_landbkgd_loader, test_waterbkgd_loader, test_landbkgd_loader  


# Hyperparameters
DEVICE = 'cuda:0'
BASE_LR = args.lr
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MOMENTUM = args.momentum
WEIGHT_DECAY = args.weight_decay
UPSAMPLE = args.upsample
STAGE_1_EPOCH = args.stage_1_epoch
RUN = args.run

STAGE_2_PATH = 'saved_models/run_%s/bs_%s_epochs_%s_lr_%s_wd_%s/'%(RUN, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY)
SAVE_PATH = STAGE_2_PATH + 'plots/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
                  
def demographic_parity(net, majority_loader, minority_loader, device):
    majority_positive_pred, minority_positive_pred = 0.0, 0.0
    majority_num_examples, minority_num_examples = 0.0, 0.0
    
    with torch.no_grad():
        for i, (features, _, _, _) in enumerate(majority_loader):
            features = features.to(DEVICE).float()
            logits = net(features)
            probas = torch.sigmoid(logits)
            predicted_labels = probas > 0.5
            majority_num_examples += features.size(0)
            majority_positive_pred += (predicted_labels == 1).sum()
            
    with torch.no_grad():
        for i, (features, _, _, _) in enumerate(minority_loader):
            features = features.to(DEVICE).float()
            logits = net(features)
            probas = torch.sigmoid(logits)
            predicted_labels = probas > 0.5
            minority_num_examples += features.size(0)
            minority_positive_pred += (predicted_labels == 1).sum()
            
    dp_gap = abs(((majority_positive_pred/majority_num_examples) - (minority_positive_pred/minority_num_examples))*100)
    dp_gap = dp_gap.item()
    
    return dp_gap     

def compute_loss_accuracy(net, loader, device):
    correct_pred, num_examples = 0.0, 0.0
    running_loss = 0.0
    cost_fn = torch.nn.BCELoss()

    with torch.no_grad():
        for i, (features, _, attributes, targets) in enumerate(loader):
            
            features = features.to(DEVICE).float()
            targets = targets.to(DEVICE).unsqueeze(1).float()

            logits = net(features)
            probas = torch.sigmoid(logits)

            primary_loss = cost_fn(probas, targets)
            running_loss += primary_loss.item()*targets.size(0)

            predicted_labels = probas > 0.5
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

    loss = running_loss/num_examples
    accuracy = (correct_pred/num_examples) * 100
    accuracy = accuracy.item()
    
    return loss, accuracy     
    
def eval_validation(model, device):
    l1, s1 = compute_loss_accuracy(model, valid_waterbirds_waterbkgd_loader, DEVICE)
    l2, s2 = compute_loss_accuracy(model, valid_waterbirds_landbkgd_loader, DEVICE)
    l3, s3 = compute_loss_accuracy(model, valid_landbirds_waterbkgd_loader, DEVICE)
    l4, s4 = compute_loss_accuracy(model, valid_landbirds_landbkgd_loader, DEVICE)
    return l1, l2, l3, l4, s1, s2, s3, s4

def eval_test(model, device):        
    l1, s1 = compute_loss_accuracy(model, test_waterbirds_waterbkgd_loader, DEVICE)
    l2, s2 = compute_loss_accuracy(model, test_waterbirds_landbkgd_loader, DEVICE)
    l3, s3 = compute_loss_accuracy(model, test_landbirds_waterbkgd_loader, DEVICE)
    l4, s4 = compute_loss_accuracy(model, test_landbirds_landbkgd_loader, DEVICE)
    return l1, l2, l3, l4, s1, s2, s3, s4

valid_waterbirds_waterbkgd_loss, valid_waterbirds_landbkgd_loss, valid_landbirds_waterbkgd_loss, valid_landbirds_landbkgd_loss = [], [], [], []
valid_waterbirds_waterbkgd_acc, valid_waterbirds_landbkgd_acc, valid_landbirds_waterbkgd_acc, valid_landbirds_landbkgd_acc = [], [], [], []

test_waterbirds_waterbkgd_loss, test_waterbirds_landbkgd_loss, test_landbirds_waterbkgd_loss, test_landbirds_landbkgd_loss = [], [], [], []
test_waterbirds_waterbkgd_acc, test_waterbirds_landbkgd_acc, test_landbirds_waterbkgd_acc, test_landbirds_landbkgd_acc = [], [], [], []

valid_dp_gaps, test_dp_gaps = [], []

valid_avg_loss, valid_avg_acc = [], []
test_avg_loss, test_avg_acc = [], []

for epoch in range(EPOCHS):
    model = models.resnet50(pretrained=True)
    d = model.fc.in_features
    model.fc = nn.Linear(d, 1)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0])
    model.to(DEVICE)    
    load_path = STAGE_2_PATH + 'epoch_%s/epoch_%s.pt'%(epoch+1, epoch+1)
    print('Loading Model Checkpoint: ', load_path)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    l1, l2, l3, l4, s1, s2, s3, s4 = eval_validation(model, DEVICE)
    valid_waterbirds_waterbkgd_loss.append(l1)
    valid_waterbirds_landbkgd_loss.append(l2)
    valid_landbirds_waterbkgd_loss.append(l3)
    valid_landbirds_landbkgd_loss.append(l4)
    valid_waterbirds_waterbkgd_acc.append(s1)
    valid_waterbirds_landbkgd_acc.append(s2)
    valid_landbirds_waterbkgd_acc.append(s3)
    valid_landbirds_landbkgd_acc.append(s4)

    v_l = (1057*l1/4795) + (56*l2/4795) + (184*l3/4795) + (3498*l4/4795)    
    v_a = (1057*s1/4795) + (56*s2/4795) + (184*s3/4795) + (3498*s4/4795)
    valid_avg_loss.append(v_l)
    valid_avg_acc.append(v_a)
    
    l1, l2, l3, l4, s1, s2, s3, s4 = eval_test(model, DEVICE)
    test_waterbirds_waterbkgd_loss.append(l1)
    test_waterbirds_landbkgd_loss.append(l2)
    test_landbirds_waterbkgd_loss.append(l3)
    test_landbirds_landbkgd_loss.append(l4)
    test_waterbirds_waterbkgd_acc.append(s1)
    test_waterbirds_landbkgd_acc.append(s2)
    test_landbirds_waterbkgd_acc.append(s3)
    test_landbirds_landbkgd_acc.append(s4)
    
    t_l = (1057*l1/4795) + (56*l2/4795) + (184*l3/4795) + (3498*l4/4795)    
    t_a = (1057*s1/4795) + (56*s2/4795) + (184*s3/4795) + (3498*s4/4795)
    test_avg_loss.append(t_l)
    test_avg_acc.append(t_a)
       
    valid_dp_gap = demographic_parity(model, valid_waterbkgd_loader, valid_landbkgd_loader, DEVICE)
    test_dp_gap = demographic_parity(model, test_waterbkgd_loader, test_landbkgd_loader, DEVICE)
    valid_dp_gaps.append(valid_dp_gap)
    test_dp_gaps.append(test_dp_gap)
    
    if (epoch+1)%1 == 0:

        np.savetxt(SAVE_PATH + "valid_waterbirds_waterbkgd_loss.txt", valid_waterbirds_waterbkgd_loss)
        np.savetxt(SAVE_PATH + "valid_waterbirds_landbkgd_loss.txt", valid_waterbirds_landbkgd_loss)
        np.savetxt(SAVE_PATH + "valid_landbirds_waterbkgd_loss.txt", valid_landbirds_waterbkgd_loss)
        np.savetxt(SAVE_PATH + "valid_landbirds_landbkgd_loss.txt", valid_landbirds_landbkgd_loss)
        np.savetxt(SAVE_PATH + "valid_waterbirds_waterbkgd_acc.txt", valid_waterbirds_waterbkgd_acc)
        np.savetxt(SAVE_PATH + "valid_waterbirds_landbkgd_acc.txt", valid_waterbirds_landbkgd_acc)
        np.savetxt(SAVE_PATH + "valid_landbirds_waterbkgd_acc.txt", valid_landbirds_waterbkgd_acc)
        np.savetxt(SAVE_PATH + "valid_landbirds_landbkgd_acc.txt", valid_landbirds_landbkgd_acc)
        
        np.savetxt(SAVE_PATH + "test_waterbirds_waterbkgd_loss.txt", test_waterbirds_waterbkgd_loss) 
        np.savetxt(SAVE_PATH + "test_waterbirds_landbkgd_loss.txt", test_waterbirds_landbkgd_loss)  
        np.savetxt(SAVE_PATH + "test_landbirds_waterbkgd_loss.txt", test_landbirds_waterbkgd_loss) 
        np.savetxt(SAVE_PATH + "test_landbirds_landbkgd_loss.txt", test_landbirds_landbkgd_loss)
        np.savetxt(SAVE_PATH + "test_waterbirds_waterbkgd_acc.txt", test_waterbirds_waterbkgd_acc) 
        np.savetxt(SAVE_PATH + "test_waterbirds_landbkgd_acc.txt", test_waterbirds_landbkgd_acc)  
        np.savetxt(SAVE_PATH + "test_landbirds_waterbkgd_acc.txt", test_landbirds_waterbkgd_acc) 
        np.savetxt(SAVE_PATH + "test_landbirds_landbkgd_acc.txt", test_landbirds_landbkgd_acc)
        
        np.savetxt(SAVE_PATH + "valid_dp_gap.txt", valid_dp_gaps)
        np.savetxt(SAVE_PATH + "test_dp_gap.txt", test_dp_gaps)

        np.savetxt(SAVE_PATH + "valid_avg_loss.txt", valid_avg_loss)
        np.savetxt(SAVE_PATH + "valid_avg_acc.txt", valid_avg_acc)
        np.savetxt(SAVE_PATH + "test_avg_loss.txt", test_avg_loss)
        np.savetxt(SAVE_PATH + "test_avg_acc.txt", test_avg_acc)
        
