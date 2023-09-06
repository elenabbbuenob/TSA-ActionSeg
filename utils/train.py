
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from statistics import mean

# Set seeds for reproducibility
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

def train(X, model, criterion, optimizer, nbatch):
    model.train()      
    frames = list(range(len(X))) #ndownsampling
    random.shuffle(frames) 
    
    num_batches= max(len(X)//nbatch,1)
    total_loss = 0
    
    for nbatch in range(num_batches):
        batch_loss =0
        optimizer.zero_grad()  
        
        with torch.set_grad_enabled(True):
            Z, mts, idx_mts = model(X)
            similarity= mts.detach().cpu().numpy()
            
            ############################################           
            for fil, idx in enumerate(idx_mts):
                #  Similarity index choice
                similarity_idx =  similarity[fil]
                # Sort similarity in descending order 
                order, dist = zip(*sorted(zip(idx_mts, similarity_idx), key = lambda x:-x[1])) 
                order = list(order)
                dist = list(dist)               
                dist.pop(0)
                
                if idx_mts[fil] in order:
                    order.remove(idx_mts[fil])                   
                # Choice of semantic positive.
                set_positive= int(len(order)*0.05)
                if order[:set_positive] !=[]:
                    peaks = order[:set_positive] 
                    positive_idx = np.random.choice(peaks, 1)[0]
                else: 
                    peaks=order[:4]
                    positive_idx = np.random.choice(order[:4], 1)[0]                                     
                        
                # Hard Negatives [mean+std, mean] and negatives [mean, median]           
                troughsHN = list(set(np.array(order)[np.array(dist) <= mean(dist) + np.std(dist)]) - set(np.array(order)[mean(dist) <= np.array(dist)]))
                
                if len(troughsHN) >= 1:
                    negatives_idx = np.random.choice(troughsHN, 1)[0]
                else:
                    troughsEN = list(set(np.array(order)[np.array(dist) <= mean(dist)]) - set(np.array(order)[np.median(dist) <= np.array(dist)]))
                    if not troughsEN:
                        troughsEN = list(set(np.array(order)[np.array(dist) <= mean(dist)]) - set(np.array(order)[max(0, mean(dist) - np.std(dist)) <= np.array(dist)]))
                        if not troughsEN:
                            troughsEN = list(np.array(order)[np.array(dist) <= mean(dist)])
                    negatives_idx = np.random.choice(troughsEN, 1)[0]

                idx_mts = list(idx_mts)
                pdf_anchor = mts[fil]
                pdf_positive = mts[idx_mts.index(positive_idx)]
                pdf_negative = mts[idx_mts.index(negatives_idx)]

                loss = criterion(pdf_anchor, pdf_positive, pdf_negative)
                
                batch_loss += loss                

            batch_loss = batch_loss/len(idx_mts)        
            batch_loss.backward()                   
            optimizer.step()   
            optimizer.zero_grad() 
            model.zero_grad() 
            total_loss = total_loss + batch_loss.item()
        
        model.eval()
        Zi, _,_ = model(X)

    return Zi, total_loss/num_batches


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['num_epoch'])) / 2        
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['lr_decay_rate'] ** steps)
    elif p['scheduler'] == 'constant':
        lr = lr
    elif p['scheduler'] == 'exponential':
        lr = lr * math.exp(-p['lr_decay_rate'] * epoch)        
    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
