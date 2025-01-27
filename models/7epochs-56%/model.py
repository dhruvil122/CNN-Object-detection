import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
    
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.fc1 =nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512,8)
        #############################
        # Initialize your network
        #############################
        
    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1,64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

        #############################
        # Implement the forward pass
        ########################
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################

        
        torch.save(self.state_dict(), 'model.pkl')

