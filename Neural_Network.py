'''
Author: Deavan
Date: 2021-04-19 14:42:59
Description: 
'''
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(28*28, 512)
        self.relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(512, 512)
        #self.relu2 = nn.ReLU()
        self.Linear3 = nn.Linear(512, 10)
        #self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.Linear1(x)
        #x = nn.ReLU(x)
        x = self.relu1(x)
        x = self.Linear2(x)
        #x = self.relu1(x)
        x = self.Linear3(x)
        #x = self.relu1(x)
        return x

model = NeuralNetwork().to(device)
print(model)

X = torch.ones(1, 28, 28, device=device)
print(X)
logits = model(X)
print(model.Linear1)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


