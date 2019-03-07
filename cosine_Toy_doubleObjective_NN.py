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

num_epochs = 20
number_size = 8
copies_per_num = 4000
DISPLAY_METRIC_INTERVAL = 100
BATCH_SIZE = 10
MODEL_PATH = "ae_nn.p"

embedding_vector_size = int(math.ceil(math.log(number_size, 2)))
use_cuda = False

# print(input_vector_size)
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
# criterion = nn.NLLLoss()

criterion = nn.CosineEmbeddingLoss()

# criterion = nn.CrossEntropyLoss()

# data_set = []
# for i in range(number_size):
#     bin_value_str = bin(i)
#     bin_value_array = [int(x) for x in bin_value_str[2:]]
#     #add necessary preceeding zeros to fix the input size
#     bin_value_array = [0.0] * (embedding_vector_size - len(bin_value_array)) + bin_value_array
#     data_set.append(bin_value_array)

data_set = []
target_set = []
for i in range(1,number_size+1): #+1 because we go from 1 to num not 0 to num-1
    one_hot_array = [0] * number_size
    one_hot_array[i-1] = 1 #-1 because the indices start at 0
    data_set += [one_hot_array]*copies_per_num
    target_set += [one_hot_array]*copies_per_num
#end for

temp_data = zip(data_set,target_set)
temp_data = list(temp_data)
shuffle(temp_data)
unzippd_data = zip(*temp_data)
unzippd_data = list(unzippd_data)
data_set = unzippd_data[0]
target_set = unzippd_data[1]

target_set = torch.tensor(target_set, dtype = torch.float)
# NEED TO PUT INDICES in the target set if using cross entropy loss

data_set = torch.tensor(data_set, dtype = torch.float)



# create MANY copies of the same data, maybe even add noise, SEE OLD code
# mimic this architecture.
#
#
#
# input_layer = Input([max_number])
# x = Dense(int(max_number/2),activation="relu")(input_layer)
# single_embedding = Dense(1,activation="linear")(x)
# # single_embedding = Dense(1,activation="linear")(input_layer)
# x = Dense(int(max_number/2),activation="hard_sigmoid")(single_embedding)
# # x = Dense(int(max_number),activation="hard_sigmoid")(x)
# output = Dense(max_number,activation="softmax")(x)
# # output = Dense(max_number,activation="hard_sigmoid")(single_embedding)



class AE_noSecond(nn.Module):
    def __init__(self):
        super(AE_noSecond,self).__init__()
        self.fc1 = nn.Linear(number_size, int(number_size/2))
        self.fc2 = nn.Linear(int(number_size/2), int(number_size/4))
        self.fc3 = nn.Linear(int(number_size / 4), int(embedding_vector_size))
        #decode
        self.fc5 = nn.Linear(int(embedding_vector_size), int(embedding_vector_size * 2))
        self.fc6 = nn.Linear(int(embedding_vector_size * 2), int(number_size/2))
        self.fc7 = nn.Linear(int(number_size/2), number_size)

        # ---------

    # def forward(self, input):
    #     curr_output = torch.sigmoid(self.fc1(input))
    #     curr_output = torch.sigmoid(self.fc2(curr_output))
    #     # curr_output = F.relu(self.fc3(curr_output))
    #     curr_output = torch.sigmoid(self.fc3(curr_output))
    #     curr_output = torch.sigmoid(self.fc5(curr_output))
    #     curr_output = torch.sigmoid(self.fc6(curr_output))
    #     curr_output = self.fc7(curr_output)
    #     # else
    #     # curr_output = F.softmax(self.fc7(curr_output))
    #     # curr_output = F.sigmoid(self.fc7(curr_output))
    #     return curr_output
    #
    #
    # # -------------------------------

    def forward(self, input):
        curr_output = F.relu(self.fc1(input))
        curr_output = F.relu(self.fc2(curr_output))
        # curr_output = F.relu(self.fc3(curr_output))
        curr_output = F.relu(self.fc3(curr_output))
        curr_output = F.relu(self.fc5(curr_output))
        curr_output = F.relu(self.fc6(curr_output))
        #if criterion is BCE with logit loss, then no sigmoid. BUT to decode still use sigmoid
        # curr_output = self.fc7(curr_output)
        #else
        curr_output = F.softmax(self.fc7(curr_output))
        # curr_output = F.sigmoid(self.fc7(curr_output))
        return curr_output

    #-------------------------------
    def get_encoding(self,input):
        curr_output = F.relu(self.fc1(input))
        curr_output = F.relu(self.fc2(curr_output))
        curr_output = F.relu(self.fc3(curr_output))

        return curr_output

    # # -------------------------------
    # def get_decoding(self,input):
    #     curr_output = F.relu(self.fc5(input))
    #     curr_output = F.relu(self.fc6(curr_output))
    #     curr_output = F.sigmoid(self.fc7(curr_output))
    #     return curr_output
    # -------------------------------

device = torch.device("cuda" if use_cuda else "cpu")

ae_nn = AE_noSecond()

try:
    print("Model was loaded")
    # pass
    # ae_nn.load_state_dict(torch.load(MODEL_PATH))
except:
    print("Model was not loaded")
    pass


ae_nn.to(device)
optimizer = optim.SGD(ae_nn.parameters(), lr = 0.01)#, momentum= 0.5)

ae_nn.train()
for i in range(num_epochs):
    print("At epoch =", i)
    cumul_loss = 0.0
    for d in range(data_set.shape[0]):
        input,target = data_set[d:d+1,:],target_set[d:d+1]
        input,target = input.to(device),target.to(device)
        output = ae_nn(input)
        loss = 1-F.cosine_similarity(output,target)
        cumul_loss += loss.data
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if d%BATCH_SIZE == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()
        if d%DISPLAY_METRIC_INTERVAL == 0:
            print("at d =",d)
            print("avg loss %f",cumul_loss/DISPLAY_METRIC_INTERVAL)
            cumul_loss = 0.0
    #end inner for
#end outer for



torch.save(ae_nn.state_dict(),MODEL_PATH)
