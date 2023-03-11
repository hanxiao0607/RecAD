import torch
import torch.nn as nn
import torch.nn.functional as F

class INet(nn.Module):
    def __init__(self, num_vars: int, k:int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device):
        super(INet, self).__init__()

        self.nets = nn.ModuleList()

        for _ in range(num_vars):
            modules = [nn.Sequential(nn.Linear(k, hidden_layer_size), nn.ReLU())]
            if num_hidden_layers > 1:
                modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
            modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, 1)))
            self.nets.append(nn.Sequential(*modules))

        self.num_vars = num_vars
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.device = device

    def forward(self, inputs: torch.Tensor):
        preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
        for i in range(self.num_vars):
            net_i = self.nets[i]
            pred_i = net_i(inputs[:, :, i])
            pred_i = F.pad(torch.reshape(pred_i, (-1, 1)), (i, self.num_vars-1-i))
            preds += pred_i
        return preds

