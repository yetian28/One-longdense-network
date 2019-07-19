#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import sys
import time
import argparse
import numpy as np
import h5py
from torch.utils.data import TensorDataset, Dataset, DataLoader
device = torch.device('cuda:0')  


parser = argparse.ArgumentParser(description='dense')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--log_every', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=8192)
parser.add_argument('--lr', type=float, default=0.002)
args = parser.parse_args(args=[])


# In[8]:



start_time = time.time()
fid = pd.HDFStore("all_rad_train_test_data_B_log_R_cpuBG_H.h5")
x_train = fid['x_train']
x_test = fid['x_test']
y_train = fid['y_train']
y_test = fid['y_test']
fid.close() #AK

print(f"time: {time.time()-start_time}[s], data is loaded from .h5 file") #AK



# In[11]:


start_time = time.time()
x_train=x_train.values
y_train=y_train.values
x_test=x_test.values
y_test=y_test.values


print(f"time: {time.time()-start_time}[s], data is converted to numpy array") #AK


# In[14]:


start_time = time.time()
x_train=x_train.astype(np.float32)
y_train=y_train.astype(np.int)
x_test=x_test.astype(np.float32)
y_test=y_test.astype(np.int)
print(f"time: {time.time()-start_time}[s], data is converted to float and int") #AK


# In[18]:


x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).type(torch.long)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).type(torch.long)


# In[19]:


y_train = F.one_hot(y_train, num_classes = 3).type(torch.float).view(-1,3)
y_test = F.one_hot(y_test, num_classes = 3).type(torch.float).view(-1,3)


# In[33]:


f = h5py.File("mytraindata.hdf5", "w")
f.create_dataset("x_train", data=x_train)
f.create_dataset("y_train", data=y_train)
f.create_dataset("x_test", data=x_test)
f.create_dataset("y_test", data=y_test)
f.close()


# In[2]:


start=time.time()
f = h5py.File("mytraindata.hdf5", "r")
x_train = f['x_train'][:,:]
x_test = f['x_test'][:,:]
y_train = f['y_train'][:,:]
y_test = f['y_test'][:,:]
f.close()
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)
print(f"time spent load data to RAM {time.time()-start}")


# In[4]:


class LongDense(nn.Module):
    def __init__(self):
        super(LongDense, self).__init__()


        self.fc = nn.Sequential(  #CheckMe!
            nn.Linear(108, 752),
            nn.ReLU6(),
            nn.Linear(752, 576),
            nn.ReLU6(),
            nn.Linear(576, 256),
            nn.ReLU6(),
            nn.Linear(256, 192),
            nn.ReLU6(),
            nn.Linear(192, 96),
            nn.ReLU6(),
            nn.Linear(96, 3),
            nn.Sigmoid()
        )


    def forward(self, x):      
        x = self.fc(x)
        return x

    

traindataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(traindataset,
        batch_size=args.batch_size, shuffle = False ,pin_memory = True) #CheckMe!
testdataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(testdataset,
        batch_size=args.batch_size, shuffle = False) #AK




longdense = LongDense().to(device)
longdense.train()
optimizer = torch.optim.Adam(longdense.parameters(), lr=args.lr) 
    
for i in range(args.num_epochs):
    start_epoch_time = time.time()  
    for j, (inputdata, labels) in enumerate(train_loader):       
        start = time.time()  
        inputdata = inputdata.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()  
        pred = longdense(inputdata)
        loss = nn.BCELoss()(pred, labels) 
        loss.backward()
        optimizer.step()  
        
        if j % args.log_every == 0:
            train_acc = torch.mean((torch.max(pred, 1)[1] == torch.max(labels, 1)[1] ).type(torch.float))
            print(f"epoch {i}, bacth_iter {j}: time spent={time.time()-start }, acc= {train_acc}, loss = {loss}")
    print(f"epoch {i}: time spent: {time.time() - start_epoch_time}")  
    

        



longdense.eval()
total = 0.
acc_sum = 0.
start = time.time()       
for i, (inputdata, labels) in enumerate(test_loader):
    inputdata = inputdata.to(device)
    labels = labels.to(device)
    with torch.no_grad(): #CheckMe!
        pred = longdense(inputdata)
    acc_sum += torch.sum((torch.max(pred, 1)[1] == torch.max(labels, 1)[1] ).type(torch.float))
    total += pred.shape[0]

acc = acc_sum / total #AK
end = time.time()
print(f'test acc = {acc}, time spent {end-start}') 



# In[6]:


start=time.time()
for j, (inputdata, labels) in enumerate(train_loader):       
    if j== 400:
        break
print(f"time spent {time.time()-start}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




