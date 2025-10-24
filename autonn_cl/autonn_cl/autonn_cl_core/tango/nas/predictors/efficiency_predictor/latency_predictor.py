'''
The currently written version uses lookup table.
It will be replaced with code of a trained predictor network later.
'''
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class LatencyPredictor:
    def __init__(self, target, target_acc, device):
        path = os.path.dirname(os.path.realpath(__file__))
        bpred_path = os.path.join(path, "trained_lat_pred/{}_{}_backbone.pt".format(target, target_acc))
        hpred_path = os.path.join(path, "trained_lat_pred/{}_{}_head.pt".format(target, target_acc))

        self.device = device
        self.b_net = Net(
            nfeat=20,
            hw_embed_on=False,
            hw_embed_dim=0,
            layer_size=64
        ).cuda()
        self.h_net = Net(
            nfeat=28,
            hw_embed_on=False,
            hw_embed_dim=0,
            layer_size=64
        ).cuda()

        self.b_net.load_state_dict(torch.load(bpred_path))
        self.h_net.load_state_dict(torch.load(hpred_path))
        self.b_net.eval()
        self.h_net.eval()

    def predict_efficiency(self, arch):
        b_feat, h_feat = arch_to_feat(arch, self.device)
        latency_b = self.b_net(b_feat)
        latency_h = self.h_net(h_feat)

        return latency_b + latency_h
    
def arch_to_feat(arch, device):
    # This function converts a backbone arch_encoding to a feature vector (20-D).
    d_list = copy.deepcopy(arch['d'])

    # convert to onehot, 5*4 = 20-D feature vector
    b_onehot = [0 for _ in range(20)]
    h_onehot = [0 for _ in range(28)]

    for i in range(4):
        tidx = int(d_list[i]) - 1
        b_onehot[i*5 + tidx] = 1
    
    for i in range(4):
        tidx = int(d_list[i+4]) - 1
        b_onehot[i*5 + tidx] = 1

    return torch.Tensor(b_onehot).to(device).float(), torch.Tensor(h_onehot).to(device).float()


class Net(nn.Module):
    """
    The base model for MAML (Meta-SGD) for meta-NAS-predictor.
    """

    def __init__(self, nfeat, hw_embed_on, hw_embed_dim, layer_size):
        super(Net, self).__init__()
        self.layer_size = layer_size
        self.hw_embed_on = hw_embed_on

        self.add_module('fc1', nn.Linear(nfeat, layer_size))
        self.add_module('fc2', nn.Linear(layer_size, layer_size))

        if hw_embed_on:
            self.add_module('fc_hw1', nn.Linear(hw_embed_dim, layer_size))
            self.add_module('fc_hw2', nn.Linear(layer_size, layer_size))
            hfeat = layer_size * 2 
        else:
            hfeat = layer_size

        self.add_module('fc3', nn.Linear(hfeat, hfeat))
        self.add_module('fc4', nn.Linear(hfeat, hfeat))

        self.add_module('fc5', nn.Linear(hfeat, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, hw_embed=None, params=None):
        # hw_embed = hw_embed.repeat(len(x), 1)
        if params == None:
            out = self.relu(self.fc1(x))
            out = self.relu(self.fc2(out))

            if self.hw_embed_on:
                hw = self.relu(self.fc_hw1(hw_embed))
                hw = self.relu(self.fc_hw2(hw))
                out = torch.cat([out, hw], dim=-1)

            out = self.relu(self.fc3(out))
            out = self.relu(self.fc4(out))
            out = self.fc5(out)

        else:
            out = F.relu(F.linear(x, params['meta_learner.fc1.weight'],
                                params['meta_learner.fc1.bias']))
            out = F.relu(F.linear(out, params['meta_learner.fc2.weight'],
                                params['meta_learner.fc2.bias']))
            
            if self.hw_embed_on:
                hw = F.relu(F.linear(hw_embed, params['meta_learner.fc_hw1.weight'],
                                    params['meta_learner.fc_hw1.bias']))
                hw = F.relu(F.linear(hw, params['meta_learner.fc_hw2.weight'],
                                    params['meta_learner.fc_hw2.bias']))
                out = torch.cat([out, hw], dim=-1)

            out = F.relu(F.linear(out, params['meta_learner.fc3.weight'],
                                params['meta_learner.fc3.bias']))
            out = F.relu(F.linear(out, params['meta_learner.fc4.weight'],
                                params['meta_learner.fc4.bias']))
            out = F.linear(out, params['meta_learner.fc5.weight'],
                                params['meta_learner.fc5.bias']) 

        return out