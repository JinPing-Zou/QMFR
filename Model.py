
import torch
from torch import nn, optim
import numpy as np

omega = 0.6

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=omega):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class UVDecomposition(nn.Module):
    def __init__(self, hidden, n_rows, n_cols):
        super().__init__()
        self.U_net = nn.Sequential(SineLayer(1, hidden*3, is_first=True),
                                   SineLayer(hidden*3, hidden*3, is_first=True),
                                   nn.Linear(hidden*3, n_rows))
        
        self.V_net = nn.Sequential(SineLayer(1, hidden*3, is_first=True),
                                   SineLayer(hidden*3, hidden*3, is_first=True),
                                   nn.Linear(hidden*3,n_cols))
        

