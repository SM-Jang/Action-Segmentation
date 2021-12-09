import os
import torch
import argparse
import random
from model import MS_TCN
from trainer import trainer
from Generator import MSTCN_gen
from evaluation import *
import pandas as pd
import pdb



def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op',               default='residual') # mul, residual, none
    parser.add_argument('--action',           default='test') # train, test, eval
    parser.add_argument('--dataset',          default="50salads")
    parser.add_argument('--split',            default='4')
    parser.add_argument('--gpu',              default=0)
    parser.add_argument('--sample_rate')
    parser.add_argument('--method')
    return parser.parse_args()


def set_path(dataset, action, split, op, method):
    vid_list_file     =  '../data/{}/splits/{}.split{}.bundle'.format(dataset, action, split)
    features_path     =  '../data/{}/features/'.format(dataset)
    gt_path           =  '../data/{}/groundTruth/'.format(dataset)
    mapping_file      =  '../data/{}/mapping.txt'.format(dataset)
    if op == 'none':  
        model_dir     =  './models/{}/split_{}/{}'.format(dataset, split, op)
        results_dir   =  './results/{}/split_{}/{}'.format(dataset, split, op)
    else:
        model_dir =  './models/{}/split_{}/{}'.format(dataset, split, method)
        results_dir   =  './results/{}/split_{}/{}'.format(dataset, split, method)
    
    if not os.path.exists(model_dir):   
        print(f"Make the model directory on {model_dir}")
        os.makedirs(model_dir)
    if not os.path.exists(results_dir): 
        print(f"Make the result directory on {results_dir}")
        os.makedirs(results_dir)
    
    return vid_list_file, features_path, gt_path, mapping_file, model_dir, results_dir


def get_actiondict(mapping_file):
    actions_dict = {}
    with open(mapping_file, 'r') as f:
        actions = f.read().split('\n')[:-1]
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
        
    return actions_dict


def set_gpu(gpu):
     ## Set GPU device ##
    GPU    = args.gpu
    device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")
    seed   = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print("Current Device:", device)
    return device

def set_params():
    num_stages    = 4
    num_layers    = 10
    num_f_maps    = 64
    features_dim  = 2048
    learning_rate = 0.0005
    num_epochs    = 50
    batch_size    = 1
    
    print(f"Number of stages :{num_stages}\nNumber of layers :{num_layers}\nNumber of f_maps :{num_f_maps}\nFeature Dimension :{features_dim}\nNumber Of Epochs :{num_epochs}\nBatch Size :{batch_size}\n")
    
    return num_stages, num_layers, num_f_maps, features_dim, learning_rate, num_epochs, batch_size
    

    
    
    
if __name__ == '__main__':
    
    args = get_argument()
    
    ## Setting ##
    device = set_gpu(args.gpu)
    num_stages, num_layers, num_f_maps, features_dim, learning_rate, num_epochs, batch_size = set_params()
    sample_rate = int(args.sample_rate)
    print(f"Sample Rate :{sample_rate}")
    vid_list_file, features_path, gt_path, mapping_file, model_dir, results_dir  = set_path(args.dataset, args.action, args.split, args.op, args.method)
    actions_dict = get_actiondict(mapping_file)
    num_classes = len(actions_dict)
    print('The number of class in {} is {}'.format(args.dataset, num_classes))
        
    trainer = trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes, args.op, args.method)
    
    
    if args.action == 'train':
        print(f'\n\t\tPre-train MS_TCN on {args.op}\n')
        batch_gen = MSTCN_gen(num_classes, actions_dict, gt_path, features_path, sample_rate, args.method)
        batch_gen.read_data(vid_list_file)
        trainer.train(model_dir, batch_gen, num_epochs, batch_size, learning_rate, device)
        
        
        

    if args.action == 'test':
        print(f'\n\t\tPredict the test dataset with MS_TCN on {args.op}\n')
        batch_gen = MSTCN_gen(num_classes, actions_dict, gt_path, features_path, sample_rate, args.method)
        batch_gen.read_data(vid_list_file)
        trainer.predict(model_dir, batch_gen, results_dir, actions_dict, batch_size, device, sample_rate)
        
        
        
            

    if args.action == 'eval':
        print(f'\n\t\tThe test scores of MS_TCN on {args.op}\n')
        result = seg_eval(args.dataset, args.split, args.op)
        result = pd.DataFrame(list(result.items()),
                           columns=['metrics', 'scores'])

        result.to_csv('./csv/control_point/{}_{}_{}_{}.csv'.format(args.dataset, args.method, args.op, args.sample_rate), columns=['metrics', 'scores'])
        print("Save the result on ./csv/control_point/{}_{}_{}_{}.csv".format(args.dataset, args.method, args.op, args.sample_rate))
