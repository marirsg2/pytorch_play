"""

A simple auto encoder for encoding and decoding a one-hot encoding or binary encoding

The learning aspect is in the bottleneck embedding. The activation function is Relu
BUT we also calculate the loss as the euclidean distance from the zero vector, which is just the L2 distance.

So ReLU and this secondary loss should push the embeddings towards [0,1]. compare the results to without this
additional loss

We make the bottleneck small to force non-trivial encoding.

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import shuffle

#synthetic dataset
# a long list of numbers which are then converted to one hot encoding or binary encoding.

num_epochs = 5
number_size = 10000
input_vector_size = int(math.ceil(math.log(number_size,2)))
use_cuda = False
# print(input_vector_size)
criterion = nn.BCELoss()

data_set = []
for i in range(number_size):
    bin_value_str = bin(i)
    bin_value_array = [int(x) for x in bin_value_str[2:]]
    #add necessary preceeding zeros to fix the input size
    bin_value_array = [0.0]*(input_vector_size-len(bin_value_array)) + bin_value_array
    data_set.append(bin_value_array)


shuffle(data_set)
target_set = torch.tensor(data_set, dtype = torch.float)
data_set = torch.tensor(data_set, dtype = torch.float)

class AE_noSecond(nn.Module):
    def __init__(self):
        super(AE_noSecond,self).__init__()
        self.fc1 = nn.Linear(input_vector_size,int(input_vector_size/2))
        self.fc2 = nn.Linear(int(input_vector_size/2),int(input_vector_size/4))
        #decode
        self.fc3 = nn.Linear(int(input_vector_size/4),int(input_vector_size/2))
        self.fc4 = nn.Linear(int(input_vector_size/2),input_vector_size)

    # -------------------------------
    def forward(self, input):
        curr_output = F.relu(self.fc1(input))
        curr_output = F.relu(self.fc2(curr_output))
        curr_output = F.relu(self.fc3(curr_output))
        curr_output = F.relu(self.fc4(curr_output))
        return curr_output
    #-------------------------------
    def get_encoding(self,input):
        curr_output = F.relu(self.fc1(input))
        curr_output = F.relu(self.fc2(curr_output))
        return curr_output
    # -------------------------------
    def get_decoding(self,input):
        curr_output = F.relu(self.fc3(input))
        curr_output = F.relu(self.fc4(curr_output))
        return curr_output
    # -------------------------------

device = torch.device("cuda" if use_cuda else "cpu")

ae_nn = AE_noSecond()
ae_nn.to(device)
optimizer = optim.SGD(ae_nn.parameters(), lr = 0.01, momentum = 0.5)
ae_nn.train()
for i in range(num_epochs):
    print("At epoch =", i)
    for d in range(data_set.shape[0]):
        input,target = data_set[d:d+1,:],target_set[d:d+1,:]
        input,target = input.to(device),target.to(device)
        optimizer.zero_grad()
        output = ae_nn(input)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        if d%200 == 0:
            print("at d =",d)
    #end inner for
#end outer for






