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

#todo try adding noise to see if it helps

#synthetic dataset
# a long list of numbers which are then converted to one hot encoding or binary encoding.

num_epochs = 20
number_size = 8
max_number = number_size
copies_per_num = 4000
DISPLAY_METRIC_INTERVAL = 100
BATCH_SIZE = 10
MODEL_PATH = "ae_nn.p"

embedding_vector_size = int(math.ceil(math.log(number_size, 2)))
use_cuda = False
criterion = nn.CrossEntropyLoss()
#---generate data
data_set = []
target_set = []
for i in range(1,number_size+1): #+1 because we go from 1 to num not 0 to num-1
    one_hot_array = [0] * number_size
    one_hot_array[i-1] = 1 #-1 because the indices start at 0
    data_set += [one_hot_array]*copies_per_num
    target_set += [i-1]*copies_per_num #the target is the index too !!
#end for
#---shuffle and put data into tensor
temp_data = zip(data_set,target_set)
temp_data = list(temp_data)
shuffle(temp_data)
unzippd_data = zip(*temp_data)
unzippd_data = list(unzippd_data)
data_set = unzippd_data[0]
target_set = unzippd_data[1]

# NEED TO PUT INDICES in the target set if using cross entropy loss
target_set = torch.tensor(target_set, dtype = torch.long)
data_set = torch.tensor(data_set, dtype = torch.float)
#---setup the NN
class Trivial_AE(nn.Module):
    def __init__(self):
        super(Trivial_AE, self).__init__()
        self.fc1 = nn.Linear(number_size, 1)
        self.fc2 = nn.Linear(1, number_size)
        self.fc3 = nn.Linear(number_size, number_size)

    def forward(self, input):
        curr_output = self.fc1(input)#no activaiton, just linear
        curr_output = F.relu(self.fc2(curr_output))
        curr_output = F.softmax(self.fc3(curr_output), dim=1)
        return curr_output
#---end nn class




device = torch.device("cuda" if use_cuda else "cpu")
ae_nn = Trivial_AE()


try:
    print("Model was loaded")
    # pass
    ae_nn.load_state_dict(torch.load(MODEL_PATH))
except:
    print("Model was not loaded")
    pass



# ae_nn.to(device)
# optimizer = optim.SGD(ae_nn.parameters(), lr = 0.003)#, momentum=0.9)
# ae_nn.train()
# for i in range(num_epochs):
#     print("At epoch =", i)
#     cumul_loss = 0.0
#     optimizer.zero_grad()
#     for d in range(data_set.shape[0]):
#         input,target = data_set[d:d+1,:],target_set[d:d+1]
#         # input,target = input.to(device),target.to(device)
#         output = ae_nn(input)
#         loss = criterion(output,target)
#         cumul_loss += loss.data
#         loss.backward()
#         if d%BATCH_SIZE == 0:
#             optimizer.step()
#             optimizer.zero_grad()
#         if d%DISPLAY_METRIC_INTERVAL == 0 and d!= 0:
#             print("at d =",d)
#             print("avg loss %f",cumul_loss/DISPLAY_METRIC_INTERVAL)
#             # print("curr loss %f",loss)
#             cumul_loss = 0.0
#     #end inner for
# #end outer for

#test it out
ae_nn.eval()

prediction_array = np.zeros(max_number)
prediction_array[0] = 1
prediction_array = torch.tensor([prediction_array],dtype = torch.float)
output = ae_nn.forward(prediction_array)
print(prediction_array)
print(output)
print("=========================================================")

prediction_array = np.zeros(max_number)
prediction_array[1] = 1
prediction_array = torch.tensor([prediction_array],dtype = torch.float)
output = ae_nn.forward(prediction_array)
print(prediction_array)
print(output)
print("=========================================================")

prediction_array = np.zeros(max_number)
prediction_array[2] = 1
prediction_array = torch.tensor([prediction_array],dtype = torch.float)
output = ae_nn.forward(prediction_array)
print(prediction_array)
print(output)
print("=========================================================")

prediction_array = np.zeros(max_number)
prediction_array[3] = 1
prediction_array = torch.tensor([prediction_array],dtype = torch.float)
output = ae_nn.forward(prediction_array)
print(prediction_array)
print(output)
print("=========================================================")

prediction_array = np.zeros(max_number)
prediction_array[4] = 1
prediction_array = torch.tensor([prediction_array],dtype = torch.float)
output = ae_nn.forward(prediction_array)
print(prediction_array)
print(output)
print("=========================================================")

prediction_array = np.zeros(max_number)
prediction_array[5] = 1
prediction_array = torch.tensor([prediction_array],dtype = torch.float)
output = ae_nn.forward(prediction_array)
print(prediction_array)
print(output)
print("=========================================================")

prediction_array = np.zeros(max_number)
prediction_array[6] = 1
prediction_array = torch.tensor([prediction_array],dtype = torch.float)
output = ae_nn.forward(prediction_array)
print(prediction_array)
print(output)
print("=========================================================")

prediction_array = np.zeros(max_number)
prediction_array[7] = 1
prediction_array = torch.tensor([prediction_array],dtype = torch.float)
output = ae_nn.forward(prediction_array)
print(prediction_array)
print(output)
print("=========================================================")



torch.save(ae_nn.state_dict(),MODEL_PATH)
