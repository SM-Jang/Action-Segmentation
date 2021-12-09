import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as  optim
import copy
import numpy as np
import sys
import os
from model import MS_TCN


    
# def get_boundary_index(labels):
#     labels = labels.squeeze().numpy()

#     boundary = []
#     cur = None
#     for i, label in enumerate(labels):
#         if cur == None:
#             cur = label

#         if cur != label:
#             cur = label
#             boundary.append(i)

#     return boundary



        



class trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, op, method):
        self.model       = MS_TCN(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce          = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse         = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.op          = op
        self.method      = method


    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device): 
        
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask, name = batch_gen.next_batch(batch_size)
                optimizer.zero_grad()
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                
                
                ## Curvature Operation ##
                curvature_tensor = batch_gen.load_curvature(name)
                batch_input = self.operation(batch_input, curvature_tensor, self.op)
                predictions = self.model(batch_input) # MS_TCN1 forward
                
                
                 ## Loss ##
                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])
                epoch_loss += loss.item()
                
                
                ## optimization ##
                loss.backward()
                optimizer.step()
                
                
                ## Train Performance ##
                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            print("[epoch %d] loss = %f | acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), float(correct) / total * 100 ) )
            
            if (epoch + 1) % 10==0:
                torch.save(self.model.state_dict(), f'{save_dir}/epoch-{epoch+1}.model')
                torch.save(optimizer.state_dict(),f'{save_dir}/epoch-{epoch+1}.opt')
                print(f"Save the model train on {self.method} on {save_dir}")
            
            

    def predict(self, model_dir, batch_gen, results_dir, actions_dict, batch_size, device, sample_rate):
        
        self.model.eval()
        with torch.no_grad():
            
            path = f'{model_dir}/epoch-{50}.model'
            self.model.to(device)
            self.model.load_state_dict(torch.load(path))
            print( f"Load the trained model weights on {path}" )
            
            while batch_gen.has_next():
                ## prediction using the trained network on method data ##
                batch_input, _, _, name = batch_gen.next_batch(batch_size)
                batch_input = batch_input.to(device)
                predictions = self.model(batch_input) # forward
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                
                
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                
                f_name = f'{results_dir}/{name[0]}'
                print("Save the prediction on {}".format(f_name))
                
                with open(f_name, 'w') as f:
                    f.write("### Frame level recognition: ###\n")
                    f.write(' '.join(recognition))

                    
                    
    def operation(self, batch_input, curvature_tensor, op):
        ## Frame-wise Reidual Operation ##
        if op in ['mul', 'residual']:
            result = []
            frames = batch_input.shape[-1]
            for frame in range(frames):
                result.append(batch_input[:,:,frame].squeeze() * curvature_tensor.squeeze()[frame])
            if op == 'mul': batch_input = torch.stack(result).T.unsqueeze(dim=0)
            if op == 'residual': batch_input = batch_input + torch.stack(result).T.unsqueeze(dim=0)
        return batch_input








from triplet import get_embedding_net
from triplet import TripletNet, TripletLoss
from triplet import TripletDataset3
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
import bezier
from bezier.hazmat.curve_helpers import evaluate_hodograph, get_curvature
from sklearn.preprocessing import minmax_scale

class triplet_trainer:
    
    def __init__(self, action, actions_dict, batch_gen, margin, lr):
        self.actions_dict = actions_dict
        if action == 'train':
            self.triplet_data, self.video_name = self.get_triplet_data(actions_dict, batch_gen)
        self.triplet_model = TripletNet(get_embedding_net(), actions_dict)
        self.optimizer = optim.Adam(self.triplet_model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                        lr_lambda=lambda epoch: 0.98 ** epoch,
                                        last_epoch=-1) 
        self.triplet_loss = TripletLoss(margin)
        

    def get_triplet_data(self, actions_dict, batch_gen):
        triplet_data = {}
        video_name = []
        batch_gen.reset()
        count = 0
        while batch_gen.has_next():
            count+=1
            batch_input, labels, _, name = batch_gen.next_batch(1)
            batch_input, labels = batch_input.squeeze(), labels.squeeze()
            video = name[0][:-4]
            video_name.append(video)
            print(f'{count}/{batch_gen.__len__()} : {video}')
            
            
            triplet_data[video] = {}
            for label in labels: # init
                triplet_data[video][label.item()] = []


            for i, label in enumerate(labels):
                for l in range(len(actions_dict)):
                    if label.item() == l:
                        triplet_data[video][l].append(batch_input[:,i])

                    
            for label in list(triplet_data[video].keys()):
                # list -> tensor
                triplet_data[video][label] = torch.stack(triplet_data[video][label])
        
        return triplet_data, video_name

    def train(self, num_epochs, batch_size, device): 
        
        self.triplet_model.to(device)
        self.triplet_model.train()
        print(self.triplet_model)
        print('=============================================Training Start!===============================================', end = '\n\n')

        total_time = 0
        loss_list = []
        for epoch in range(num_epochs):

            epoch_loss = 0
            start = time.time()

            for video in self.video_name:
                triplet_dataset = TripletDataset3(self.triplet_data, self.actions_dict, video)
                triplet_loader  = DataLoader(triplet_dataset, batch_size, shuffle=True)

                for i, (anchor, positive, negative, anchor_label) in enumerate(triplet_loader):

                    data = (Variable(anchor.squeeze(), requires_grad=True).to(device), 
                            Variable(positive.squeeze(), requires_grad=True).to(device), 
                            Variable(negative.squeeze(), requires_grad=True).to(device))


                    outputs = self.triplet_model(*data)
                    self.optimizer.zero_grad()

                    loss = self.triplet_loss(*outputs)


                    epoch_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()


            total_time += (time.time() - start)
            print("Epoch {}/{}\tDuration {:.2f}s \tLearning Rate {:.5f}\t Loss {:.5f} ".format(epoch+1, num_epochs, 
                                                                                          time.time()-start, 
                                                                                          self.scheduler.get_lr()[0], 
                                                                                          epoch_loss))

#             if (epoch +1) % 100 == 0 :
#                 print(f'save the model on ./models/triplet/triplet_{dataset}_{epoch}')
#                 torch.save({
#                             'epoch': epoch,
#                             'model_state_dict': self.triplet_model.state_dict(),
#                             'optimizer_state_dict': self.optimizer.state_dict(),

#                             }, f'./models/triplet/triplet_{dataset}_{epoch}')
                
            if (epoch +1) % 100 == 0 :
                print(f'save the model on ./models/ablation/triplet_{dataset}_{epoch}')
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.triplet_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),

                            }, f'./models/ablation/triplet_{dataset}_{epoch}')
                
            
    def save_curvature(self, dataset, batch_gen, step, device):
        path = f'./models/triplet/triplet_{dataset}'
        pretrained_weight = torch.load(path)['model_state_dict']
        self.triplet_model.load_state_dict(pretrained_weight)
        self.triplet_model.to(device)
        i=0
        while batch_gen.has_next():
            i+=1
            batch_input, batch_target, mask, name = batch_gen.next_batch(1)
            name = name[0][:-4]
            
            print(i, name, end='\t')
            
            
            batch_input = batch_input.T.squeeze().to(device)
            frames = batch_input.shape[0]
            index = np.arange(0, frames, step)

            embedding = self.triplet_model.get_embedding(batch_input).T.detach().cpu().numpy()
            embedding = embedding[:, index]
            curve = bezier.Curve.from_nodes(embedding)

            kappa=[]
            for s in range(frames): # calcuate the frames number of curvature 
                t = s / frames
                tangent_vec = curve.evaluate_hodograph(t)
                kappa.append(get_curvature(embedding, tangent_vec, t))
            kappa = minmax_scale(kappa)

            print(f'Save the curvature of step:{step} on ../data/{dataset}/triplet_{step}/{name}.npy')
            np.save(f'../data/{dataset}/triplet_{step}/{name}.npy', kappa)
            
            
            
            
            
            
            
            