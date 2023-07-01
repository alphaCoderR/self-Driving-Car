# AI for Self Driving Car

# Importing the libraries

import numpy as np
import os
import random
import torch
import torch.nn as nn                   # Contains all tools for neural networks
import torch.nn.functional as F         # Contains all different functions
import torch.optim as optim             # Contains optimizers
import torch.autograd as autograd       # Tensors along with gradient are stored in a Variable
from torch.autograd import Variable


# Creating the architecture of the Neural Networks
# Class Network is inheriting all properties of nn.Module class
class Network(nn.Module): 
    
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

class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        
    def push_event_to_memory(self,event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample_picker(self,batch_size):
        samples=zip(*random.sample(self.memory,batch_size))
        # if [(1,2,3),(4,5,6)] then zip()=> {(1,4),(2,5),(3,6)}
        # This bundles all states, rewards , actions of events individually
        return map(lambda x: Variable(torch.cat(x,0)), samples)
        # Now all the tensors along with their gradients are stored in pytorch's Variable
        

# Implementing the Deep Q Learning Model

class Dqn():
    
    def __init__(self, input_size, nb_actions, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_actions)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # The last_state is a vector of 5 dimensions [3 sensor signals, + orientation, - orientation]
        # But for pytorch to process it it needs to be a torch tensor + it also need to contain one 
        # more fake dimension that corresponds to the batch
        # This is because the network can only accept batch of input observations
        # The fake dimension is created by unsqueeze() & it is kept as the beginning dimension 
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        probability = F.softmax(self.model(Variable(state, volatile = True))*7) # Temperature = 7
        # state is a torch tensor so it is wise to convert it into torch Variable for better computations
        # volatile = True checks that the the gradient value is not used during computations
        # so the end value will be a torch variable that doesn't contains he gradients values in it
        # This will save us some memory and will improve the performance
        # The temperature parameter would allow us to modulate how the neural network will be sure of
        # which action to play. T->0(action least selected) & T>0 (action has some probability to get selected)
        action = probability.multinomial()
        return action.data[0,0]
    
    def learn(self, ):
        
    
