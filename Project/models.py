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

        self.M = self.M_tensor(dims)

    def M_tensor(self, dims):
      if dims == 1:
        return torch.Tensor([[0, -1],[1, 0]])

      elif dims == 2:
        return torch.Tensor([[0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

    # forward pass with differentiating
    def forward(self, x):
      x = torch.autograd.Variable(x, requires_grad=True)
      H = self.model(x)
      out = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

      return H, out @ self.M
      


# simple autoencoder
class MLP_autoencoder(nn.Module):
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

    self.tanh = nn.Tanh()

  def encode(self, x):
    x = self.tanh(self.fcs1(x))
    x = self.tanh(self.fcs2(x))
    x = self.tanh(self.fcs3(x))
    return self.fcs4(x)

  def decode(self, x):
    x = self.tanh(self.fcs5(x))
    x = self.tanh(self.fcs6(x))
    x = self.tanh(self.fcs7(x))
    return self.fcs8(x)

  def forward(self, x):
    latent = self.encode(x)
    out = self.decode(latent)
    return out

# HNN autoencoder
class PixelHNN(nn.Module):
  def __init__(self, dims):
    super(PixelHNN, self).__init__()
    self.autoencoder = MLP_autoencoder(dims=dims)
    self.hnn = HNN(dims=1)

  def model_diff(self, x):
    return self.hnn.forward(x)[1]

  def encode(self, x):
    return self.autoencoder.encode(x)

  def decode(self, latent):
    return self.autoencoder.decode(latent)

  def forward(self, x):
    latent = self.encode(x)
    out = latent + self.model_diff(latent)
    return self.decode(out)
