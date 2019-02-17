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
from random import shuffle

#synthetic dataset
# a long list of numbers which are then converted to one hot encoding or binary encoding.

number_size = 10000
input_vector_size = int(math.ceil(math.log(number_size,2)))
# print(input_vector_size)

data_set = []
for i in range(number_size):
    bin_value_str = bin(i)
    bin_value_array = [int(x) for x in bin_value_str[2:]]
    #add necessary preceeding zeros to fix the input size
    bin_value_array = [0]*(input_vector_size-len(bin_value_array)) + bin_value_array
    data_set.append(bin_value_array)


shuffle(data_set)
data_set = torch.tensor(data_set)

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




