

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
device = torch.device('cuda:0')  #CheckMe!

parser = argparse.ArgumentParser(description='dense')
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--log_every', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=8192)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args(args=[])



class RealDataset(Dataset):
  
    def __init__(self, x_train, y_train):
       
        self.x_train = x_train
        self.y_train = y_train


    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = self.x_train[idx]
        y = self.y_train[idx]
        y=torch.tensor(y, dtype = torch.long)
        return x, y
    




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


start_time = time.time()
fid = pd.HDFStore("all_rad_train_test_data_B_log_R_cpuBG_H.h5")
x_train = fid['x_train']
x_test = fid['x_test']
y_train = fid['y_train']
y_test = fid['y_test']

fid.close() #AK

print(f"time: {time.time()-start_time}[s], data is loaded from .h5 file") #AK


start_time = time.time()
x_train=x_train.values
y_train=y_train.values
x_train=x_train.astype(np.float32)
y_train=y_train.astype(np.int)
x_test=x_test.values
y_test=y_test.values
x_test=x_test.astype(np.float32)
y_test=y_test.astype(np.int)

print(f"time: {time.time()-start_time}[s], data is converted to numpy array") #AK




traindataset = RealDataset(x_train=x_train, y_train=y_train) #CheckMe!
train_loader = torch.utils.data.DataLoader(traindataset,
        batch_size=args.batch_size, shuffle=True) #CheckMe!
testdataset = RealDataset(x_train=x_test, y_train=y_test)
test_loader = torch.utils.data.DataLoader(testdataset,
        batch_size=args.batch_size, shuffle=False) #AK




longdense = LongDense().to(device)
longdense.train()
optimizer = torch.optim.Adam(longdense.parameters(), lr=args.lr) #AK
    
for i in range(args.num_epochs):
   
    start_epoch_time = time.time()  #AK
    for j, (inputdata, labels) in enumerate(train_loader):
        start = time.time()  
        inputdata = inputdata.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  
        pred = longdense(inputdata)
        loss = nn.CrossEntropyLoss()(pred, labels.view(-1))  #CheckMe!
        loss.backward()
        optimizer.step()    
        end = time.time()
        if j % args.log_every == 0:
            train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
            #print("epoch",str(i),"iter",str(j),"acc=",train_acc)
            print(f"epoch {i}, bacth_iter {j}: time spent={end-start}, acc= {train_acc}, loss = {loss}")
    print(f"epoch {i}: time spent: {time.time() - start_epoch_time}")  #AK

        
longdense.eval()
total = 0.
acc_sum = 0.
start = time.time()       
for i, (inputdata, labels) in enumerate(test_loader):
    inputdata = inputdata.to(device)
    labels = labels.to(device)
    with torch.no_grad(): #CheckMe!
        pred = longdense(inputdata)
    acc_sum += torch.sum((torch.max(pred, 1)[1] == labels).type(torch.float))
    total += pred.shape[0]

acc = acc_sum / total #AK
end = time.time()
print("acc=", acc) 

