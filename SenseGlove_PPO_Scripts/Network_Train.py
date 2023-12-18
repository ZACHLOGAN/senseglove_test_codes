import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset


data = torch.from_numpy(np.genfromtxt('network_values.csv',delimiter=',')).float().unsqueeze(-1)

positions = data[:,0]
motor_output = data[:,1]
print

class MyData(Dataset):
    def __init__(self):
        self.sequences = positions
        self.target = motor_output
        
    def __getitem__(self,i):
        return self.sequences[i], self.target[i]
    
    def __len__(self):
        return len(self.sequences)
    
class Neural_Model(nn.Module):
    def __init__(self):
        super(Neural_Model, self).__init__()

        self.layer1 = nn.Linear(1, 15)
        self.layer2 = nn.Linear(15, 1)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.activation(x)
        x = self.layer2(x)
        return x

training_set = MyData()
loader = torch.utils.data.DataLoader(training_set, batch_size=20)
print(loader)
model = Neural_Model().to('cpu')
loss_function = nn.MSELoss()
optimizer = Adam(model.parameters(), lr = 0.01)

n_epochs = 100

for epochs in range(n_epochs):
    for inputs,target in loader:
        print(target)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
    

torch.save(model.state_dict(),'./my_network.pth')

