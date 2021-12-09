import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model import MS_TCN, SS_TCN
from Generator import MSTCN_gen
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import bezier
from bezier.hazmat.curve_helpers import evaluate_hodograph, get_curvature
from sklearn.preprocessing import minmax_scale



def read_file(path, actions_dict):
    result = []
    with open(path, 'r') as f:
        label = f.read().split('\n')[:-1]
    for action in label:
        result.append(actions_dict[action])
    return result


def get_embedding_net():
    in_dim = 2048
    embedding_net = nn.Sequential(
        nn.Linear(in_dim, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),



        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),

        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),

        nn.Linear(128, 3),
    )
    return embedding_net
    
    
    
    
# Anchor를 random 순서로 뽑아서 진행하도록 하자
class TripletDataset1(Dataset):
    def __init__(self, batch_input, labels):
        self.batch_input = batch_input.squeeze()
        self.labels = labels.squeeze().numpy()
        
        self.action_frame_dict = self.get_action_frame_dict()
        self.anchor_index_set  = list(self.action_frame_dict.keys())
        random.shuffle(self.anchor_index_set)
        
        self.index_range = list(range(len(self.action_frame_dict)))
        
    def __len__(self):
        return len(self.action_frame_dict)
    
    def get_action_frame_dict(self):
        """
        같은 action을 갖는 segment를 하나의 set으로 형성
        index: 한 비디오 안에서의 set 위치
        action_frame_dict[index][0]: index 위치의 set의 label
        action_frame_dict[index][1]: index 위치의 set의 모든 frame들
        """
        result = {}
        
        cur = None
        frame_idx_list = []
        idx = 0
        for i, label in enumerate(self.labels):
            if cur == None:
                cur = label
            if cur == label:
                frame_idx_list.append(i)
            if cur != label:
                result[idx] = [cur, frame_idx_list]
                cur = label
                
                idx += 1
                
                frame_idx_list = []
                frame_idx_list.append(i)
        result[idx] = [cur, frame_idx_list]
                
        return result
    
    def divide_5set(self, index):
        set_size = len(self.action_frame_dict[index][1])
        
        if set_size < 5:
            frame_sets = [self.action_frame_dict[index][1] for i in range(5)]
            
            
        else:
        
        
            size = int((set_size) / 5)

            frame_sets = [self.action_frame_dict[index][1][ x * size : (x+1) * size] for x in range(4) ]
            frame_sets.append(self.action_frame_dict[index][1][4*size:])
        
        return frame_sets
    
    def __getitem__(self, index):
        """
        middle of index: anchor zone(k)
        index: positive set(k) 
        index +- 1: negative set(k/2+k/2)
        """
#         import pdb; pdb.set_trace()
        index = self.anchor_index_set[index] # anchor의 randomness
        k=100

        anchor_label = self.action_frame_dict[index][0]
        
        current_set = self.divide_5set(index)

        # anchor k sampling
        anchor_set = current_set[2]
        anchors = random.choices(anchor_set, k=1)
        anchors = anchors * k
#         anchors = random.choices(anchor_set, k=k)

        
        # positive k sampling
        p =  0.2
        positive_set = random.choices(current_set,
                                     weights = [4*p, 2*p, p, 2*p, 4*p],
                                     k=k)

        positives = [ random.choices(positive) for positive in positive_set]
        positives = sum(positives, [])
        # negative k sampling
        if index == 0:
            negative_set = self.divide_5set(index+1)
            negative_set = negative_set[:2]
            negative_set = sum(negative_set, [])
            negatives = random.choices(negative_set, k=k)

            
        elif index  == self.__len__() - 1:
            negative_set = self.divide_5set(index-1)
            negative_set = negative_set[-2:]
            negative_set = sum(negative_set, [])
            negatives = random.choices(negative_set, k=k)

            
        else:
            negative_set = self.divide_5set(index+1)
            negative_set = negative_set[:2]
            negative_set = sum(negative_set, [])
            negatives = random.choices(negative_set, k=int(k/2))
            
            negative_set = self.divide_5set(index-1)
            negative_set = negative_set[-2:]
            negative_set = sum(negative_set, [])
            negatives = negatives + random.choices(negative_set, k=int(k/2))


        return self.batch_input[:,anchors], self.batch_input[:,positives], self.batch_input[:,negatives], anchor_label
    
class TripletDataset2(Dataset):
    def __init__(self, triplet_data, actions_dict):
        self.triplet_data = triplet_data
        self.actions_dict = actions_dict
        self.I2A = {v: k for k, v in self.actions_dict.items()}
        
        
        self.action_frame_dict = self.get_action_frame_dict()
        self.index_set_list =list(range(self.__len__()))
        random.shuffle(self.index_set_list)
        
    def __len__(self):
        return len(self.actions_dict) # 50salds -> 19
    
        
    
    def get_action_frame_dict(self):
        """
        모든 비디오를 탐색하여
        label별로 frame을 분류
        """
        result = {}
        for action in range(len(self.actions_dict)):
            result[action] = [] # init

        for video in self.triplet_data.keys(): 
            for i, label in enumerate(self.triplet_data[video][1]):
                # 지정된 비디오의 모든 label
                for action in range(len(self.actions_dict)): # 1~19
                    # 어떤 label에 해당하는지 확인
                    if (label.item() == action):
                        result[action].append(self.triplet_data[video][0][:,i]) # 데이터
                        break
        return result
    
    def __getitem__(self, index): # 50salads 1~19
        # anchor: 1 point * k
        # positive: k points (same label)
        # negative: k points (diff label)
        
        
        k     = 32
        index = self.index_set_list[index]
        anchor_label = self.I2A[index]
        
        anchor = random.choice(self.action_frame_dict[index])
        anchor = torch.repeat_interleave(anchor.reshape(1,2048), k, dim=0)
        
        
        positive = random.choices(self.action_frame_dict[index], k = k)
        positive = torch.stack(positive)
        
        
        negative = []
        s = 0
        for i in self.index_set_list[:-1]:
            if i == index: continue
            negative.append( random.choices(  self.action_frame_dict[i], k = k //( self.__len__()-1  ))   ) 
            s+= k //( self.__len__()-1  )

        negative.append( random.choices(self.action_frame_dict[self.index_set_list[-1]], k = k  - s ) )
        negative = sum( negative, [])
        negative = torch.stack(negative)

        return anchor, positive, negative, anchor_label
    
class TripletDataset3(Dataset):
    def __init__(self, triplet_data, actions_dict, video):
        self.video = video
        self.triplet_data = triplet_data
        self.actions_dict = actions_dict
        self.I2A = {v: k for k, v in self.actions_dict.items()}
        
        
        
        
        self.label_set_list = list(self.triplet_data[self.video].keys())
        random.shuffle(self.label_set_list)
        
    def __len__(self):
        return len(self.triplet_data[self.video]) # 50salds -> 19
    

    def __getitem__(self, label): # 50salads 1~19

        """
        anchor -> intersection with video and label
        positive -> same label, diff video(col)
        negative -> same video, diff label(row)
        
        """
        k     = 32
        label = self.label_set_list[label]
        anchor_label = self.I2A[label]
        
        
        
        anchor_zone = self.triplet_data[self.video][label]
        anchor = random.choices(self.triplet_data[self.video][label], k = k)
        anchor = torch.stack(anchor)
        
        
        positive_zone = []
        for video in list(self.triplet_data.keys()):
            positive_zone.append(self.triplet_data[self.video][label])
        positive = torch.cat(positive_zone)
        positive = torch.stack(random.choices(positive, k = k))
        
        
        
        negative_zone = []
        for label in list(self.triplet_data[self.video].keys()):
            negative_zone.append(self.triplet_data[self.video][label])
        negative = torch.cat(negative_zone)
        negative = torch.stack(random.choices(negative, k = k))
        

        return anchor, positive, negative, anchor_label    
    
    
class TripletNet(nn.Module):
    def __init__(self, embedding_net, actions_dict):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        self.actions_dict = actions_dict
        self.I2A = {v: k for k, v in self.actions_dict.items()}

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1) # anchor
        output2 = self.embedding_net(x2) # positive
        output3 = self.embedding_net(x3) # negative
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
    
    
    
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin


    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(dim=1) # Frame(embedding points)의 개수 만큼 positive 
        distance_negative = (anchor - negative).pow(2).sum(dim=1) # Frame(embedding points)의 개수 만큼 negative 
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()
    

    