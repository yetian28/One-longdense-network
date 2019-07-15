#!/usr/bin/env python
# coding: utf-8

# ## Preparation

# In[40]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import sys
import time
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda:0') 

parser = argparse.ArgumentParser(description='dense')
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--log_every', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args(args=[])


# ## Model

# In[21]:


torch.cuda.current_device()
torch.cuda.get_device_name(0)
torch.cuda.is_available()


# In[33]:


class LongDense(nn.Module):
    def __init__(self):
        super(LongDense, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(58, 752),
            nn.BatchNorm1d(752),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(752, 576),
            nn.BatchNorm1d(576),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(576, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Linear(96, 3)
        )


    def forward(self, x):
        x = self.fc(x)
        return x


# ## Load data

# In[15]:


fid = pd.HDFStore("all_rad_train_test_data_C_log_cpuBG_H.h5")
x_train = fid['x_train']
x_test = fid['x_test']
y_train = fid['y_train']
y_test = fid['y_test']
x_train=torch.tensor(x_train.values)
x_test=torch.tensor(x_test.values)
y_train=torch.tensor(y_train.values, dtype=torch.long)
y_test=torch.tensor(y_test.values, dtype=torch.long)


# In[41]:


class RealDataset(Dataset):
  
    def __init__(self, x_train, y_train):
       
        self.x_train = x_train
        self.y_train = y_train


    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = self.x_train[idx]
        y = self.y_train[idx]
        return x, y
    
traindataset = RealDataset(x_train=x_train, y_train=y_train)
train_loader = torch.utils.data.DataLoader(traindataset,
        batch_size=args.batch_size, shuffle=True)
testdataset = RealDataset(x_train=x_test, y_train=y_test)
test_loader = torch.utils.data.DataLoader(testdataset,
        batch_size=args.batch_size, shuffle=True)


# ## Main function

# In[35]:


#cuda
def main():
    longdense = LongDense()
    longdense.cuda()
    longdense.train()
    optimizer = torch.optim.Adam(longdense.parameters(), lr=args.lr, 
            betas=(0.95, 0.95), eps=1e-2)
    
    for i in range(args.num_epochs):
        startep=time.time()
        for j, (inputdata, labels) in enumerate(train_loader):
            start = time.time()
            
            inputdata = inputdata.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()  
            pred = longdense(inputdata)
            loss = nn.CrossEntropyLoss()(pred, labels.view(-1))
            loss.backward()
            optimizer.step()    
           # train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
            end = time.time()
            if (j) % args.log_every == 0:
                train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
                display = 'epoch=' + str(i) +                       '\tacc=%.4f' % (train_acc) +                       '\ttime=%.2fit/s' % (1. / (end - start))
                print(display)
        endep=time.time()
        print('\ttime=%.2fit/s' % (1. / (endep - startep)))
        
    
    longdense.eval()
    total = 0.
    acc_sum = 0.
    for i, (inputdata, labels) in enumerate(test_loader):
        start = time.time()
            
        inputdata = inputdata.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            pred = longdense(inputdata)
        acc_sum += torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
        total += pred.shape[0]
        acc = acc_sum / total
        end = time.time()
        if (i) % args.log_every == 0:
            display = '\tacc=%.4f' % (acc) +                       '\ttime=%.2fit/s' % (1. / (end - start))
            print(display)    


# In[42]:


main()


# In[11]:


longdense.eval()
total = 0.
acc_sum = 0.
for i, (inputdata, labels) in enumerate(test_loader):
    start = time.time()
            
    inputdata = inputdata.cuda()
    labels = labels.cuda()
    with torch.no_grad():
        pred = longdense(inputdata)
    acc_sum += torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
    total += pred.shape[0]
    acc = acc_sum / total
    end = time.time()
    if (i) % args.log_every == 0:
        display = '\tacc=%.4f' % (acc) +                       '\ttime=%.2fit/s' % (1. / (end - start))
        print(display)    


# In[10]:


test_loader = torch.utils.data.DataLoader(testdataset,
        batch_size=4096, shuffle=True)


# In[19]:


torch.cuda.empty_cache()

