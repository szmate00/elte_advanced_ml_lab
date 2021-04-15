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

# simple autoencoder
class MLP_autoencoder(torch.nn.Module):
  def __init__(self, dims):
    super(MLP_autoencoder, self).__init__()
    self.fcs1 = torch.nn.Linear(dims, 200)
    self.fcs2 = torch.nn.Linear(200, 200)
    self.fcs3 = torch.nn.Linear(200, 200)
    self.fcs4 = torch.nn.Linear(200, 2)

    self.fcs5 = torch.nn.Linear(2, 200)
    self.fcs6 = torch.nn.Linear(200, 200)
    self.fcs7 = torch.nn.Linear(200, 200)
    self.fcs8 = torch.nn.Linear(200, dims)

    def encode(self, x):
    x = nn.Tanh((self.fcs1(x))
    x = x + nn.Tanh(self.fcs2(x) 
    x = x + nn.Tanh(self.fcs3(x))
    return self.fcs4(x)

    def decode(self, x):
    x = nn.Tanh((self.fcs5(x))
    x = x + nn.Tanh(self.fcs6(x) 
    x = x + nn.Tanh(self.fcs7(x))
    return self.fcs8(x)

    def forward(self, x):
    x = self.encode(x)
    out = self.decode(x)
    return out

# HNN autoencoder
class PixelHNN(torch.nn.Module):
  def __init__(self, dims):
    super(MLP_autoencoder, self).__init__()
    self.fcs1 = torch.nn.Linear(dims, 200)
    self.fcs2 = torch.nn.Linear(200, 200)
    self.fcs3 = torch.nn.Linear(200, 200)
    self.fcs4 = torch.nn.Linear(200, 2)

    self.fcs5 = torch.nn.Linear(2, 200)
    self.fcs6 = torch.nn.Linear(200, 200)
    self.fcs7 = torch.nn.Linear(200, 200)
    self.fcs8 = torch.nn.Linear(200, dims)

    def encode(self, x):
    x = nn.Tanh((self.fcs1(x))
    x = x + nn.Tanh(self.fcs2(x) 
    x = x + nn.Tanh(self.fcs3(x))
    return self.fcs4(x)

    def decode(self, x):
    x = nn.Tanh((self.fcs5(x))
    x = x + nn.Tanh(self.fcs6(x) 
    x = x + nn.Tanh(self.fcs7(x))
    return self.fcs8(x)

    def forward(self, x):
    x0 = self.encode(x)
    out = # x0 + dx0/dt kéne h ez legyen, ide kéne a model deriválás
    return self.decode(out)
