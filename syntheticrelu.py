

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import linalg as LA

class LinearCNN(nn.Module):
    def __init__(self, input_dim, out_channel, patch_num):
        super(LinearCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, out_channel*2, int(input_dim/patch_num), int(input_dim/patch_num))
        self.out_channel = out_channel

    def forward(self, x):
        x = self.conv1(x)
        x = torch.pow(x, 1)
        x = torch.nn.functional.relu(x)
        x = torch.mean(x,2)
        output = torch.stack([torch.sum(x[:,:self.out_channel],1), torch.sum(x[:,self.out_channel:],1)]).transpose(1,0)
        return output

def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.normal_(m.weight.data, mean=0.0, std= 0.01)

"""### Data Generation"""

DATA_NUM = 400
CLUSTER_NUM = 1
PATCH_NUM = 2
PATCH_LEN = 50
Noiselevel = 1.0
bmu = 1
flip = 0.1

features = torch.zeros(CLUSTER_NUM, PATCH_LEN)
pos = 0
for i in range(CLUSTER_NUM):
    features[i,pos] = bmu
    pos+=1

data = []
labels = []

for i in range(DATA_NUM*2):
    y = np.random.choice([-1,1], 1)[0] 
    k = torch.randint(0, CLUSTER_NUM, (1,))
    xi = torch.tensor(np.random.normal(0, Noiselevel, size=(PATCH_LEN))) 
    # make it orthogonal 
    # xi[:CLUSTER_NUM] = 0
    
    x = torch.stack([features[k][0]*y, xi])
    
    # random permutation
    idx = torch.randperm(len(x))
    x = x[idx].flatten()
    y = np.random.binomial(1,flip,1)[0]*y


    data.append(x)
    labels.append(y)

data = torch.stack(data)
print(data.shape)

labels = torch.tensor(labels)
labels[labels==-1] = 0
print(labels.shape)

training_data = data[:DATA_NUM,:].unsqueeze(1).float()
test_data = data[DATA_NUM::].unsqueeze(1).float()
print(training_data.shape)
print(training_data.shape, test_data.shape)

training_labels = labels[:DATA_NUM]
test_labels = labels[DATA_NUM:]
print(training_labels.shape, test_labels.shape)

"""### Training"""

import torch.optim as optim

def train(model, criterion, data, labels, optimizer, epochs):

    for epoch in range(epochs):  

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        # if epoch%500 == 0:   
        #     print('Epoch %d --- loss: %.3f' %
        #             (epoch + 1, loss.item()))
        #     #test(model, criterion, test_data, test_labels, num_epochs)

    # print('Finished Training')
    
def train_with_Margin(model, criterion, data, labels, optimizer, epochs):
    Largest = []
    Smallest = []
    trainR = []
    testR = []
    testtR = []
    for epoch in range(epochs):  
        funcf = model(data)[:, 1] -  model(data)[:, 0]
        yfuncf = funcf*(labels - 0.5)*2
        Largest.append(yfuncf.max().detach())
        Smallest.append(yfuncf.min().detach())
        testR.append(test_ACC(model, criterion, test_data, test_labels, epoch))
        testtR.append(test(model, criterion, test_data, test_labels, epoch))
        trainR.append(test(model, criterion, data, labels, epoch))

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        
    return Largest, Smallest, testR, testtR, trainR

        
        # if epoch%500 == 0:   
        #     print('Epoch %d --- loss: %.3f' %
        #             (epoch + 1, loss.item()))
        #     #test(model, criterion, test_data, test_labels, num_epochs)

    # print('Finished Training')


def test(model, criterion, data, labels, epochs):
    correct = 0
    
    with torch.no_grad():
        outputs = model(data)
    #     predicted = torch.max(outputs.data, 1).indices
    #     correct += (predicted == labels).sum().item()
    # outputt = 100*correct / data.shape[0]
    outputt = criterion(outputs, labels)
    # print('Accuracy of the network on the %d test images: %.4f %%' % (data.shape[0],
    #     outputt))
    return outputt

def test_ACC(model, criterion, data, labels, epochs):
    correct = 0
    
    with torch.no_grad():
        outputs = model(data)
        predicted = torch.max(outputs.data, 1).indices
        correct += (predicted == labels).sum().item()
    outputt = 1 - correct / data.shape[0]
    # print('Accuracy of the network on the %d test images: %.4f %%' % (data.shape[0],
    #     outputt))
    return outputt

# Single Run Codes
random.seed(205348320) 
num_epochs = 100
DATA_NUM = 20
TEST_DATA_NUM = 1000

CLUSTER_NUM = 1
PATCH_NUM = 2
# PATCH_LEN = 500
PATCH_LEN = 100
Noiselevel = 1.0
bmu = 5

data = []
labels = []
for i in range(DATA_NUM + TEST_DATA_NUM):
  features = torch.zeros(CLUSTER_NUM, PATCH_LEN)
  pos = 0
  for p in range(CLUSTER_NUM):
      features[p,pos] = 1
      pos+=1
  y = np.random.choice([-1,1], 1)[0] 
  k = torch.randint(0, CLUSTER_NUM, (1,))
  xi = torch.tensor(np.random.normal(0, Noiselevel, size=(PATCH_LEN))) 
  # orthogonal
  # xi[:CLUSTER_NUM] = 0    
  x = torch.stack([features[k][0]*bmu*y, xi])    
  # random permutation
  idx = torch.randperm(len(x))
  x = x[idx].flatten()
  y = np.random.choice([-1,1], 1, p = [flip, 1-flip])[0] *y
  data.append(x)
  labels.append(y)
data = torch.stack(data)
labels = torch.tensor(labels)
labels[labels==-1] = 0

training_data = data[:DATA_NUM].unsqueeze(1).float()
test_data = data[DATA_NUM:].unsqueeze(1).float()
training_labels = labels[:DATA_NUM]
test_labels = labels[DATA_NUM:]
my_mixture = LinearCNN(2*PATCH_LEN, 10, PATCH_NUM) #input_dim, out_channel (m), cluter_num, patch_num
# my_mixture.apply(initialize_weights)

criterion = torch.nn.CrossEntropyLoss() 
optimizer = optim.SGD(my_mixture.parameters(), lr= 0.1, momentum=0, weight_decay=0)
Lar, Sam, testr, testtr, trainr = train_with_Margin(my_mixture, criterion, training_data, training_labels, optimizer, num_epochs)

plt.plot(Lar, label = "Largest yf Number")
plt.plot(Sam, label = "smallest yf Number (Margin)")
plt.plot(np.array(Lar) - np.array(Sam), label = "Difference (Largest - Smallest)")

plt.title("Sensitivity of Different Instance",fontsize=15)
plt.xlabel("Iteration",fontsize=13)
plt.ylabel("Function Value",fontsize=13)
# plt.yscale("log")
plt.legend()
plt.show()

plt.plot(trainr, label = "training loss")
plt.plot(testr, label = "test error")
# plt.plot(testtr, label = "test loss")

plt.title("training loss and test error change",fontsize=15)
plt.xlabel("Iteration",fontsize=13)
plt.ylabel("Value",fontsize=13)
# plt.yscale("log")
plt.legend()
plt.show()

random.seed(205348320) 
num_epochs = 100
DATA_NUM = 20
TEST_DATA_NUM = 1000

CLUSTER_NUM = 1
PATCH_NUM = 2
# PATCH_LEN = 500
PATCH_LEN = 100
Noiselevel = 1.0
strength = []
test_acc = []
train_acc = []


Data_train = np.random.rand(100, 100)
Data_test = np.random.rand(100, 100)
for s in range(100):
  bmu = 0.1*s 
  strength.append(bmu)
  for t in range(100):
    print(100*s + t)
    test1 = 0
    train1 = 0
    PATCH_LEN = 100 + 10*t
    for j in range(10):
      data = []
      labels = []
      for i in range(DATA_NUM + TEST_DATA_NUM):
        features = torch.zeros(CLUSTER_NUM, PATCH_LEN)
        pos = 0
        for p in range(CLUSTER_NUM):
            features[p,pos] = 1
            pos+=1
        y = np.random.choice([-1,1], 1)[0] 
        k = torch.randint(0, CLUSTER_NUM, (1,))
        xi = torch.tensor(np.random.normal(0, Noiselevel, size=(PATCH_LEN))) 
        # orthogonal
        # xi[:CLUSTER_NUM] = 0    
        x = torch.stack([features[k][0]*bmu*y, xi])    
        # random permutation
        idx = torch.randperm(len(x))
        x = x[idx].flatten()
        y = np.random.choice([-1,1], 1, p = [flip, 1-flip])[0] *y
        data.append(x)
        labels.append(y)
      data = torch.stack(data)
      labels = torch.tensor(labels)
      labels[labels==-1] = 0

      training_data = data[:DATA_NUM].unsqueeze(1).float()
      test_data = data[DATA_NUM:].unsqueeze(1).float()
      training_labels = labels[:DATA_NUM]
      test_labels = labels[DATA_NUM:]
      my_mixture = LinearCNN(2*PATCH_LEN, 10, PATCH_NUM) #input_dim, out_channel (m), cluter_num, patch_num
      criterion = torch.nn.CrossEntropyLoss() 
      optimizer = optim.SGD(my_mixture.parameters(), lr= 0.1, momentum=0, weight_decay=0)
      train(my_mixture, criterion, training_data, training_labels, optimizer, num_epochs)
      train1 += test(my_mixture, criterion, training_data, training_labels, num_epochs)
      test1 += test_ACC(my_mixture, criterion, test_data, test_labels, num_epochs)
    # test_acc.append(test1/10)
    # train_acc.append(train1/10)
    # print(train1/10)
    # print(test1/10)
    Data_train[t][s] = train1/10
    Data_test[t][s] = test1/10

data[TEST_DATA_NUM::].size()

data[:DATA_NUM].size()

data[DATA_NUM:].size()

data[:TEST_DATA_NUM].size()

import seaborn as sns

# Data_test1 = Data_test[50:100, 0:50]
# Data_train1 = Data_train[50:100, 0:50]

A1 =  np.multiply(range(100), 0.1)
# A1 =  range(1, 11)
A2 =  np.multiply(range(1, 101), 10)
xlabels = ['{:3.1f}'.format(x) for x in A1]
# ylabels = ['{:3.1f}'.format(y) for y in A2]
ylabels = A2
ax = sns.heatmap(Data_test, xticklabels = xlabels, yticklabels = ylabels, cmap = "YlGnBu", vmax = 1)
# ax = sns.heatmap(Data_test, cmap = "YlGnBu")
ax.invert_yaxis()

ax.set_title("Test Error Heatmap")
ax.set_xlabel('Strength of the Signal')
ax.set_ylabel('dimension')
ax.set_xticks(ax.get_xticks()[::10])
ax.set_xticklabels(xlabels[::10])
ax.set_yticks(ax.get_yticks()[::10])
ax.set_yticklabels(ylabels[::10])
plt.show()

np.multiply(range(100), 0.1)**4

from google.colab import drive
drive.mount('/content/gdrive')

np.savetxt('data_test100.csv', Data_test)
np.savetxt('data_train100.csv', Data_train)
!cp data_test100.csv "gdrive/My Drive/"
!cp data_train100.csv "gdrive/My Drive/"

# Data_test1 = Data_test[50:100, 0:50]
# Data_train1 = Data_train[50:100, 0:50]

A1 =  np.multiply(range(100), 0.1)
# A1 =  range(1, 11)
A2 =  np.multiply(range(1, 101), 10)
xlabels = ['{:3.1f}'.format(x) for x in A1]
# ylabels = ['{:3.1f}'.format(y) for y in A2]
ylabels = A2
ax = sns.heatmap(Data_test> 0.2, xticklabels = xlabels, yticklabels = ylabels, cmap = "YlGnBu", vmax = 1)
# ax = sns.heatmap(Data_test, cmap = "YlGnBu")
ax.invert_yaxis()

ax.set_title("Test Error Heatmap")
ax.set_xlabel('Strength of the Signal')
ax.set_ylabel('dimension')
ax.set_xticks(ax.get_xticks()[::10])
ax.set_xticklabels(xlabels[::10])
ax.set_yticks(ax.get_yticks()[::10])
ax.set_yticklabels(ylabels[::10])
plt.show()

# Data_test1 = Data_test[50:100, 0:50]
# Data_train1 = Data_train[50:100, 0:50]

A1 =  np.multiply(range(100), 0.01)
A2 =  range(100, 1100)
xlabels = ['{:3.1f}'.format(x) for x in A1]
# ylabels = ['{:3.1f}'.format(y) for y in A2]
ylabels = A2
ax = sns.heatmap(Data_train, xticklabels = xlabels, yticklabels = ylabels, cmap = "YlGnBu", vmax = 1)
# ax = sns.heatmap(Data_test, cmap = "YlGnBu")
ax.invert_yaxis()

ax.set_title("Training Loss Heatmap")
ax.set_xlabel('SNR')
ax.set_ylabel('Training Data Number')
ax.set_xticks(ax.get_xticks()[::10])
ax.set_xticklabels(xlabels[::10])
ax.set_yticks(ax.get_yticks()[::10])
ax.set_yticklabels(ylabels[::10])
plt.show()

plt.suptitle('n = 100, d = 400')
plt.plot(strength,test_acc, label = 'test nonlinear')
plt.plot(strength,train_acc, label = 'training nonlinear')
plt.xlabel('Signal Ratio (mu divided by the noise std)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()