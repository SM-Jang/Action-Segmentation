import torch.nn as nn



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
