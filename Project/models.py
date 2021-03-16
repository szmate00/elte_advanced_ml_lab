import numpy as np
import torch
import torch.nn as nn


# baseline multilayer perceptron
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(2, 200),
                                  nn.Tanh(),
                                  nn.Linear(200, 200),
                                  nn.Tanh(),
                                  nn.Linear(200, 2))    
        
    def forward(self, x):
        out = self.model(x)
        
        return out

# differentiable hamiltonian neural network
class HNN(nn.Module):
    def __init__(self):
        super(HNN, self).__init__()
        self.model = nn.Sequential(nn.Linear(2, 200),
                                  nn.Tanh(),
                                  nn.Linear(200, 200),
                                  nn.Tanh(),
                                  nn.Linear(200, 1))    
        
    def forward(self, x):
        x = torch.autograd.Variable(x, requires_grad=True)
        out = torch.autograd.grad(self.model(x).sum(), x, create_graph=True)[0]

        return out @ torch.Tensor([[0, -1],[1, 0]])