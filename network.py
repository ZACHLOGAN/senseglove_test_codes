"""

This file contains the neural network module used for creating the actor network in the ppo algorithm
This neural network module can also be used for creating the critic network used in the ppo algorithm

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#this neural network will be the same basis for actor and critic networks
class Neural_Network(nn.Module):
    #setting parameters for creating the neural network
    def __init__(self, in_dim, out_dim):
        """

        Parameters:
            in_dim - the dimension of the inputs as an integer
            out_dim - the dimension of the outputs as an integer
            
        Return:
            None
            
        """
        #intialize the class for the neural network to be usable in other python codes
        super(FeedForwardNN, self).__init__()

        #define the the network's number of layers and number of nodes in each layer
        #this structure must be the same as the network used to learn intial motor activation policy
        self.input_layer = nn.Linear(in_dim, 32)
        self.hlayer1 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, out_dim)

    #run the forward pass on the neural network
    def forward(self, obs):
        """

            Parameters:
                obs - observations being passed as inputs

            Return:
                output - the output of the forward pass on the neural network
                
        """
        #if the observations are passes as numpy array, convert to PyTorch Tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).unsqueeze(-1)

        #compute activations and layer values in the network (number of activations is equal to number of layers - 1)
        activation1 = F.relu(self.input_layer(obs))
        activation2 = F.relu(self.hlayer1(activation1))
        output = self.output_layer(activation2)

        return output
