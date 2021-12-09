import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable


import random
import bezier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from main import set_path
from Generator import MSTCN_gen
from triplet import TripletDataset1, TripletLoss
from triplet import TripletNet, get_embedding_net




def get_actiondict(mapping_file):
    actions_dict = {}
    with open(mapping_file, 'r') as f:
        actions = f.read().split('\n')[:-1]
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
        
    return actions_dict


def embeddings3d_plot(embeddings, batch_target, name, step):
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(f"Embedding Plot: {name}")
    ax = fig.add_subplot(projection='3d')
    labels = np.array(batch_target.squeeze())
    frames = embeddings.shape[1]
    index = np.arange(0, frames, step)


    for i in index:
        c1 = colors[labels[i]]
        ax.scatter(
            embeddings[0, i],    # x-coordinates.
            embeddings[1, i],    # y-coordinates.
            embeddings[2, i],    # z-coordinates.
            s = 10,
            color=c1

        )
#     plt.savefig(f'./fig/ablation/1/{name}.jpeg')
    plt.show()
    plt.close()
    
    
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
dataset      = '50salads'
mapping_file = f'../data/{dataset}/mapping.txt'
actions_dict = get_actiondict(mapping_file)
num_classes  = len(actions_dict)
batch_size   = 1
op           = 'residual'
method       = 'triplet'
step         = 10
margin       = 10.
lr           = 0.005
num_epochs   = 100
colors       = sns.color_palette("Set3", len(actions_dict)) ## 색상 지정


if dataset  == "50salads": sample_rate = 2
else: sample_rate=1


vid_list_file, features_path, gt_path, _, _, _ = set_path(dataset, 'train', '4', op, method)
batch_gen = MSTCN_gen(num_classes, actions_dict, gt_path, features_path, sample_rate, method)
batch_gen.read_data(vid_list_file)
batch_gen.has_next()






while batch_gen.has_next():    
    batch_input, batch_target, _, name = batch_gen.next_batch(1)
    name        = name[0][:-4]
    triplet_dataset = TripletDataset1(batch_input, batch_target)
    triplet_loader = DataLoader(triplet_dataset, batch_size)
    print(f"Create triplet model for {name}")
    embedding_net = get_embedding_net()
    triplet_model = TripletNet(embedding_net, actions_dict).to(device)    
    
    
    #scheduler
    triplet_loss = TripletLoss(margin)
    optimizer    = optim.Adam(triplet_model.parameters(), lr=lr)
    scheduler    = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                    lr_lambda=lambda epoch: 0.98 ** epoch,
                                    last_epoch=-1) 
    
    triplet_model.train()
    loss_list = []
    print('=============================================Training Start!===============================================', end = '\n\n')
    for epoch in range(num_epochs):
        video_loss = 0
        idx = 0
        optimizer.zero_grad()
        for i, (batch_anchor, batch_positive, batch_negative, l) in enumerate(triplet_loader):

            batch_anchor = torch.transpose(batch_anchor.squeeze(), 1, 0)
            batch_positive = torch.transpose(batch_positive.squeeze(), 1, 0)
            batch_negative = torch.transpose(batch_negative.squeeze(), 1, 0)

            data = (Variable(batch_anchor, requires_grad=True).to(device), 
                    Variable(batch_positive, requires_grad=True).to(device), 
                    Variable(batch_negative, requires_grad=True).to(device))




            outputs = triplet_model(*data)
            set_loss = triplet_loss(*outputs)
            video_loss += set_loss

        video_loss.backward()
        optimizer.step()

        # Epoch 단위
        if (epoch +1) % 10 == 0:
            print("Epoch[{}/{}] | Video Learning_rate {:4f} | Video Loss {:4f}".format(epoch+1, num_epochs, scheduler.get_lr()[0], video_loss.item())) 
            scheduler.step()

    print(f'save the {name} model on ./models/ablation/triplet1_{dataset}_{name}')
    torch.save({
                'epoch': epoch,
                'model_state_dict': triplet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),

                }, f'./models/ablation/triplet1_{dataset}_{name}')
        
    print(f"Visualization on {name}")
    batch_input.squeeze_()
    frames      = batch_input.shape[0]
    index       = np.arange(0, frames, step)
    embeddings = triplet_model.get_embedding(batch_input.T.to(device)).T.detach().cpu().numpy()
    embeddings3d_plot(embeddings, batch_target, name, step)

                





