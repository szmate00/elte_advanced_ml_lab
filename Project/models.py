import numpy as np
import torch
import torch.nn as nn


# baseline multilayer perceptron
class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(2 * dims, 200),
                                  nn.Tanh(),
                                  nn.Linear(200, 200),
                                  nn.Tanh(),
                                  nn.Linear(200, 2 * dims))    
        
    def forward(self, x):
        out = self.model(x)
        
        return out

# differentiable hamiltonian neural network
class HNN(nn.Module):
    def __init__(self, dims):
        super(HNN, self).__init__()
        self.model = nn.Sequential(nn.Linear(2 * dims, 200),
                                  nn.Tanh(),
                                  nn.Linear(200, 200),
                                  nn.Tanh(),
                                  nn.Linear(200, 1))
        
    def forward(self, x):
        out = self.model(x)

        return out
