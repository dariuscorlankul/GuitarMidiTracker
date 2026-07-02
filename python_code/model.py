import torch
from torch import nn


class GuitarNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(188, 100),
            nn.ReLU(),
            nn.Linear(100, 94)
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)


model = GuitarNN()




