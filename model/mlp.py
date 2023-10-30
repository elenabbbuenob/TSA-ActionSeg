import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(1)


class MLP(nn.Module):
    def __init__(self, ninput, noutput, nfeat, nhidden, mt, ms, mtsinput, ndownsampling, h=1):
        super(MLP, self).__init__()
        # Mts layers
        self.weights= nn.Parameter(torch.rand(mtsinput, 1))        
        self.mt = mt
        self.ms = ms 
        self.pool = StochasticPool(mtsinput, ndownsampling) 
         
        # Input layers         
        self.fc_input = nn.Linear(ninput, nfeat)
        self.batch_input= nn.BatchNorm1d(nfeat)

        # Hidden layers
        fc = []
        for i in range(0, nhidden):
            fc.append(nn.Linear(nfeat, nfeat))
        self.fc_hidden = nn.ModuleList(fc)
        self.batch_hidden = nn.BatchNorm1d(nfeat)
        
        # Output layer
        self.fc_output = nn.Linear(nfeat, noutput)
        self.batch_output = nn.BatchNorm1d(noutput)        

    def forward(self, x):          
        # Training part
        w = F.sigmoid(self.weights)    
        mts= w*self.mt +(1-w)*self.ms  
        mts_downsampling, mts_idx= self.pool(mts)    
            
        # MLP part
        fc_x = mts@x 
        x = F.relu(self.batch_input(self.fc_input(fc_x))) 
        
        # MLP part
        for fc in self.fc_hidden:
            x = F.relu(self.batch_hidden(fc(x)))
        x = F.relu(self.batch_output(self.fc_output(x)))         
        return F.normalize(x, dim=1), mts_downsampling, mts_idx
    
class StochasticPool(nn.Module):    
    def __init__(self, ninput, ndownsampling):
        super(StochasticPool, self).__init__()
        self.ndownsampling = ndownsampling 
        self.ninput = ninput  
   
    def forward(self, mts):
        # because multinomial likes to fail on GPU when all values are equal 
        # Try randomly sampling without calling the get_random function a million times
        mts = mts.view(mts.size(0),mts.size(1))         
        idx,_ = torch.sort(torch.Tensor(np.random.choice(range(0, self.ninput), self.ndownsampling, replace=False)).type(torch.cuda.LongTensor))
        mts_downsampling = mts[idx].T[idx] # remove frames idx per column and per rows
        
        return mts_downsampling, idx.detach().cpu().numpy()
    
