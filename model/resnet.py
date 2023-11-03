import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from modules import *
from .aligner import Aligner
from utils import *
import time

class ResNet(nn.Module):
    def __init__(self, hp):
        super(ResNet, self).__init__()
        hdim = hp.hidden_dim
        self.proj = Conv1d(hp.n_mel_channels, hdim, 1)
        
        self.ResBlocks = nn.ModuleList([ResBlock(hdim, hp.kernel_size, stride=2, sigma_square=hp.sigma_square, ver_f=hp.ver_f),
                                        ResBlock(hdim, hp.kernel_size, stride=1),
                                        ResBlock(hdim, hp.kernel_size, stride=2, sigma_square=hp.sigma_square, ver_f=hp.ver_f),
                                        ResBlock(hdim, hp.kernel_size, stride=1),
                                        ResBlock(hdim, hp.kernel_size, stride=2, sigma_square=hp.sigma_square, ver_f=hp.ver_f),
                                        ResBlock(hdim, hp.kernel_size, stride=1)])
        
        self.linear = nn.Linear(hdim, hp.n_classes)
        self.register_buffer('ids', torch.arange(4096))
        
    def forward(self, x, x_lengths=None):
        if x_lengths is not None:
            x_mask = self.get_mask_from_lengths(x_lengths, x.size(-1))
        else:
            x_lengths = torch.LongTensor([x.size(-1)], device=x.device)
            x_mask = self.get_mask_from_lengths(x_lengths, x.size(-1))
            
        x = self.proj(x, x_mask)
        score_loss_total = 0
        
        alignment_list = []
        for resblock in self.ResBlocks:
            x, x_mask, x_lengths, alignment, score_loss = resblock(x, x_mask, x_lengths)
            alignment_list.append(alignment)
            score_loss_total = score_loss_total + score_loss
            
        x = (x*x_mask.unsqueeze(1)).sum(-1) / x_lengths.unsqueeze(-1)
        
        return F.log_softmax(self.linear(x), dim=-1), alignment_list, score_loss_total

    def get_mask_from_lengths(self, lengths, max_len):
        '''
        return bool_type mask that looks like below:
        [[1,1,1,1,0],
         [1,1,0,0,0],
         [1,1,1,1,1]]
        '''
        mask = self.ids[:max_len] < lengths.unsqueeze(-1)
        return mask


##### Align Stride #####
class ResBlock(nn.Module):
    def __init__(self, hdim, kernel_size, stride, sigma_square=5.0, ver_f=False):
        super(ResBlock, self).__init__()
        self.stride=stride
        
        self.bn1 = nn.BatchNorm1d(hdim)
        self.conv1 = Conv1d(hdim, hdim, kernel_size, padding=kernel_size//2, bias=False)
        
        self.bn2 = nn.BatchNorm1d(hdim)
        self.conv2 = Conv1d(hdim, hdim, kernel_size, padding=kernel_size//2)
        
        if self.stride!=1:
            self.aligner = Aligner(hdim, stride, sigma_square, ver_f=ver_f)
            self.conv3 = Conv1d(hdim, hdim, 1)
            
    def forward(self, x, x_mask, x_lengths):
        if self.stride==1:
            y = self.conv1(F.relu(self.bn1(x)*x_mask.unsqueeze(1)), x_mask)
            y = self.conv2(F.relu(self.bn2(y)*x_mask.unsqueeze(1)), x_mask)
            return x+y, x_mask, x_lengths, None, x.mul(0).mean()
        
        elif self.stride!=1:
            y = self.conv1(F.relu(self.bn1(x)*x_mask.unsqueeze(1)), x_mask)
            x = self.conv3(x, x_mask)
            
            if self.aligner.ver_f==False:
                y, y_mask, y_lengths, alignment, score_loss = self.aligner(y, x_mask, x_lengths)
                x = torch.bmm(alignment, x.transpose(1,2)).transpose(1,2)
                
            else:
                y, y_mask, y_lengths, z_zeros, indices, x_weights, alignment, score_loss = self.aligner(y, x_mask, x_lengths)
                x = z_zeros.scatter_add(2, indices, x*x_weights)
                
            y = self.conv2(F.relu(self.bn2(y)*y_mask.unsqueeze(1)), y_mask)
            
            return x+y, y_mask, y_lengths, alignment[-1, :y_lengths[-1], :x_lengths[-1]], score_loss
