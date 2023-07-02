       
# AI for Self Driving Car

# Importing the libraries

 
import numpy as np
import random
import os
import torch
import torch.nn as nn                            # Contains all tools for neural networks
import torch.nn.functional as F                  # Contains all different functions
import torch.optim as optim                      # Contains optimizers
import torch.autograd as autograd                 # Tensors along with gradient are stored in a Variable
from torch.autograd import Variable


# Creating the architecture of the Neural Networks
# Class Network is inheriting all properties of nn.Module class
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 40)
        self.fc2 = nn.Linear(40, nb_action)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        # if [(1,2,3),(4,5,6)] then zip()=> {(1,4),(2,5),(3,6)}
        # This bundles all states, rewards , actions of events individually
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
        # Now all the tensors along with their gradients are stored in pytorch's Variable

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
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
        probs = F.softmax(self.model(Variable(state, volatile = True))*70) # T=100
        # state is a torch tensor so it is wise to convert it into torch Variable for better computations
        # volatile = True checks that the the gradient value is not used during computations
        # so the end value will be a torch variable that doesn't contains he gradients values in it
        # This will save us some memory and will improve the performance
        # The temperature parameter would allow us to modulate how the neural network will be sure of
        # which action to play. T->0(action least selected) & T>0 (action has some probability to get selected)
        # When T is less it gives the agent to explore its environment  more where as if T is of great value then
        # The agent becomes more goal oriented
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()                             # In every loop optimizers have to re-initialized
        td_loss.backward(retain_graph = True)                  # Backpropagation is done, retain_variables improves back-propagation
        self.optimizer.step()                                  # Weight Updation is done
    
    def update(self, reward, new_signal):
        # This function receives the state and the reward from the game class in map.py & then process it
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # To convert float value to tensor "torch.Tensor" is used
        # To convert interger to tensor "torch.LongTensor" is used
        # An action is choosen
        action = self.select_action(new_state)
        if len(self.memory.memory) > 150:
            # If the memory is full then a random sample batch is choosen
            # & the batch is fed to the neural network & then the learning is done
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(150)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
         # All the variables are updated with new values
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        # It is made sure that the reward window should not contain more than 1000 elements
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    # This function calculates the mean of all the reward scores.
    # Thus giving us a final score
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    # Function to save the model state because in the last state of the model
    # we have our most updated weights & optimizers are also associated with the weights
    # so they are also saved
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    # Function to load the saved state to the neural network
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
