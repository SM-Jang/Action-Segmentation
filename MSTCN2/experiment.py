import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import random

import bezier as b
import os
import matplotlib.pyplot as plt
from model import Trainer
from batch_gen import BatchGenerator
from bezier.hazmat.curve_helpers import evaluate_hodograph, get_curvature
import easydict
from model import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

num_epochs = 50
features_dim = 2048
bz = 1
lr = 0.00005

num_layers_PG = 11
num_layers_R = 10
num_R = 3
num_f_maps = 64
split="4"

dataset = '50salads'


sample_rate = 1
if dataset == "50salads":
    sample_rate = 2


vid_list_file = "../data/"+dataset+"/splits/train.split"+split+".bundle"
vid_list_file_tst = "../data/"+dataset+"/splits/test.split"+split+".bundle"
features_path = "../data/"+dataset+"/features/"
gt_path = "../data/"+dataset+"/groundTruth/"

mapping_file = "../data/"+dataset+"/mapping.txt"

model_dir = "./experiment/models/"+dataset+"/split_"+split
results_dir = "./experiment/results/"+dataset+"/split_"+split

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
    
num_classes = len(actions_dict)



trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, dataset, split)
batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
batch_gen.read_data(vid_list_file)
trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)
trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)