import numpy as np
import random
import torch

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

def similaridad_norm(X, param=0.01, cenorm=0): 
    N = X.size()[0]
    dim=X.size()[1]          
    Z = torch.mm(X,X.t())        
    djj = torch.sqrt(torch.diag(Z))*torch.ones(1,N).t().cuda()+1e-16
    Z = torch.div(1 - torch.div(Z,torch.mul(djj,djj.t())) , dim)
    G = torch.exp(torch.mul(Z,-1/param))
    
    if cenorm==1: # normalize full matrix
        G=G / torch.sum(G)
    else: # row-wise cross-entropy
        z = torch.sum(G,0)
        G = torch.div(G,z.repeat(N,1))
    
    return G   
