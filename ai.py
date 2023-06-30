# AI for Self Driving Car

# Importing the libraries

import numpy as np
import os
import random
import torch
import torch.nn as nn                   # Contains all tools for neural networks
import torch.nn.functional as F         # Contains all different functions
import torch.optim as optim             # Contains optimizers
import torch.autograd as autograd       # Tensors are stored in a Variable
from torch.autograd import Variable


# Creating the architecture of the Neural Networks

class Network(nn.Module): # Class Network is inheriting all properties of nn.Module class
    
    def __init__(self,input_size,actions):
        super(Network,self).__init__()
        self.input_size=input_size
        self.actions=actions
        self.connection1=nn.Linear(input_size,40)
        self.connection2=nn.Linear(40, actions)
        
    def forward_propagation(self,state):
        hidden_layer=F.relu(self.connection1(state))
        q_values=self.connection2(hidden_layer)
        return q_values
        

# Implementing Experience Replay

