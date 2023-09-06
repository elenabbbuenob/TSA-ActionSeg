import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn import metrics 


class kl_loss(nn.Module):
    def __init__(self, p):
        self.kl_triplet_loss = nn.KLDivLoss(reduction= 'batchmean')
    def forward(self, pdf_anchor, pdf_positive, pdf_negative):
        positive = self.kl_triplet_loss(pdf_positive, pdf_anchor)
        negative = self.kl_triplet_loss(pdf_negative, pdf_anchor)
        
        # Calculate L2 loss
        l2_loss = torch.sum(pdf_anchor**2) + torch.sum(pdf_positive**2) 
        
        return F.relu(negative - positive) + 0.02 * l2_loss * 0.25 
