import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from main import set_path
from model import MS_TCN, SS_TCN
from Generator import MSTCN_gen
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset


from trainer import triplet_trainer
import time


import bezier
from bezier.hazmat.curve_helpers import evaluate_hodograph, get_curvature
from sklearn.preprocessing import minmax_scale



def get_actiondict(mapping_file):
    actions_dict = {}
    with open(mapping_file, 'r') as f:
        actions = f.read().split('\n')[:-1]
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
        
    return actions_dict



## Set GPU device ##
GPU    = 0
device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")
seed   = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
print("Device:", device)


## Set Basic Things ##
action       = 'train'
dataset      = '50salads'
mapping_file = f'../data/{dataset}/mapping.txt'
actions_dict = get_actiondict(mapping_file)
num_classes  = len(actions_dict)
batch_size   = 1
split        = '4'
op           = 'residual'
method       = 'triplet'
step         = 8

margin = 10.
lr = 0.005
num_epochs = 500
if dataset  == "50salads": sample_rate = 2
else: sample_rate=1






if action == 'train':
    print(f'\n\t\tTrain triplet network on {op}\n')
    vid_list_file, features_path, gt_path, _, _, _ = set_path(dataset, 'train', '4', op, method)
    batch_gen = MSTCN_gen(num_classes, actions_dict, gt_path, features_path, sample_rate, method)
    batch_gen.read_data(vid_list_file)
    trainer = triplet_trainer(action, actions_dict, batch_gen, margin, lr)
    trainer.train( num_epochs, batch_size, device)

if action == 'curvature':
    print(f'\n\t\tLoad pretrained triplet network\n')

    if not os.path.isdir(f'../data/{dataset}/triplet_{step}'): 
                         os.mkdir(f'../data/{dataset}/triplet_{step}')
    vid_list_file, features_path, gt_path, _, _, _ = set_path(dataset, 'test', '4', op, method)
    batch_gen = MSTCN_gen(num_classes, actions_dict, gt_path, features_path, sample_rate, method)
    batch_gen.read_data(vid_list_file)
    trainer = triplet_trainer(action, actions_dict, batch_gen, margin, lr)
    trainer.save_curvature(dataset, batch_gen, step, device)

