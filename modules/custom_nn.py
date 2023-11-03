import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)

    def forward(self, x, x_mask=None):
        if x_mask is None:
            output = super(Conv1d, self).forward(x)
        else:
            output = super(Conv1d, self).forward(x) * x_mask.unsqueeze(1)
            
        return output
