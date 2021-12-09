# Step1.
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

import bezier
from bezier.hazmat.curve_helpers import evaluate_hodograph, get_curvature
from sklearn.preprocessing import minmax_scale


class MSTCN_gen(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, method):
        
        self.list_of_examples  = list()
        self.index             = 0
        self.num_classes       = num_classes
        self.actions_dict      = actions_dict
        self.gt_path           = gt_path
        self.features_path     = features_path
        self.sample_rate       = sample_rate
        self.method            = method
        
    
    
    def read_data(self, vid_list_file):
        with open(vid_list_file, 'r') as f:
            self.list_of_examples = f.read().split('\n')[:-1]
        random.shuffle(self.list_of_examples)
    
        
    def __len__(self):
        return len(self.list_of_examples)
    
    
    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)
        
        
    def has_next(self):
        if self.index < self.__len__(): return True
        else: return False
        
    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        
        return batch_input_tensor, batch_target_tensor, mask, batch
        
        
    def load_curvature(self, name):
        name = name[0][:-4]
        

        k_path = self.features_path + '../{}/'.format(self.method) + name + '.npy'
        curvature_tensor = np.load(k_path)
        curvature_tensor = torch.tensor(curvature_tensor, dtype=torch.float32)
        return  curvature_tensor
    

            