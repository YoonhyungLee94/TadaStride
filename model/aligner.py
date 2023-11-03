import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *
from utils import *
import numpy as np
import time

class Aligner(nn.Module):
    def __init__(self, hdim, stride, sigma_square=5.0, ver_f=False):
        super(Aligner, self).__init__()
        self.stride = stride
        self.score = nn.Conv1d(hdim, 1, 1, bias=False)
        self.score.weight.data.zero_()
        
        self.sigma_square = sigma_square
        self.ver_f=ver_f
        self.register_buffer("indices", torch.arange(4096).unsqueeze(0).unsqueeze(0))
        self.register_buffer("z", torch.zeros(1, hdim, 4096))
        
    def forward(self, x, x_mask, x_lengths):
        B, D, T = x.size()
        L = math.ceil(T/self.stride)
        z_lengths = torch.ceil(x_lengths / self.stride).long()
        z_mask = x_mask[:, ::self.stride]
        
        score = self.score(x).exp().squeeze(1) * x_mask
        cum_score = torch.cumsum(score, dim=-1)  # B, T
        cum_score_norm = (((cum_score-cum_score[:, 0:1]) / (cum_score[:,-1:]-cum_score[:,0:1])) * (z_lengths.unsqueeze(-1) - 1)).unsqueeze(-1)
        score_loss = self.get_score_loss(cum_score_norm, x_mask, x_lengths)
        
        if self.ver_f==False:
            distance = -self.sigma_square*((self.indices[:,:,:L]-cum_score_norm)**2).transpose(1, 2)  # B, T/2, T
            distance = distance.masked_fill(~x_mask.unsqueeze(1), -np.inf)
            alignment = F.softmax(distance, dim=-1) * z_mask.unsqueeze(-1)
                
            z = torch.bmm(alignment, x.transpose(1, 2)).transpose(1, 2)

            return z, z_mask, z_lengths, alignment, score_loss
        
        elif self.ver_f==True:
            dist_target = torch.round(cum_score_norm).long()
            distance = -self.sigma_square*((dist_target-cum_score_norm)**2)  # B, T, 1

            alignment = distance.new_ones(B, L, T) * (-1e9)
            alignment = alignment.scatter(1, dist_target.transpose(1, 2), distance.transpose(1, 2))
            alignment = alignment.masked_fill(~x_mask.unsqueeze(1), -np.inf)
            alignment = F.softmax(alignment, dim=-1)
            alignment = alignment * z_mask.unsqueeze(-1)
            
            x_weights = alignment.sum(1, keepdim=True)
            indices = dist_target.squeeze(-1).unsqueeze(1).repeat(1, x.size(1), 1)
            z = self.z[:, :, :L].repeat(B, 1, 1)
            z = z.scatter_add(2, indices, x*x_weights)

            return z, z_mask, z_lengths, self.z[:, :, :L].repeat(B, 1, 1), indices, x_weights, alignment, score_loss

    def get_score_loss(self, cum_score_norm, x_mask, x_lengths):
        score_loss = (cum_score_norm[:, 1:]-cum_score_norm[:, :-1]).squeeze(-1)
        score_loss = ((torch.relu(score_loss-1.0) * x_mask[:, 1:]).sum(dim=-1) / (x_lengths-1)).mean()
        return score_loss