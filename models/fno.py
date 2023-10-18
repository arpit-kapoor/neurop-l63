from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from .conv import SpectralConv3d


class FNO3d(nn.Module):
    """
    FNO contains 5 Fourier interal operator layers
    """

    def __init__(self, modes1, modes2, modes3, width, 
                 n_blocks=5, in_channel=4, device='cpu'):
        super(FNO3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.n_blocks = n_blocks
        self.upscale = nn.Linear(in_channel, self.width).to(device)
        self.spectral_conv3d = [SpectralConv3d(self.width, self.width,
                                               self.modes1, self.modes2, 
                                               self.modes3).to(device) for _ in range(self.n_blocks)]
        self.linear = [nn.Conv1d(self.width, self.width, 1).to(device) for _ in range(self.n_blocks)]

        self.downscale = nn.Linear(self.width, 128).to(device)
        self.output = nn.Linear(128, 1).to(device)

    def forward(self, x):
        """
        FNO forward pass for single batch
        
        input shape: (batchsize, x1, y2, x3, p=1)
        output shape: (batchsize, x1, y2, x3, p=1)
        """
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        
        x = self.upscale(x)
        x = x.permute(0, 4, 1, 2, 3)

        for i in range(self.n_blocks):
            x1 = self.spectral_conv3d[i](x)
            x2 = self.linear[i](x.view(batchsize, self.width, -1)).view(batchsize, self.width,
                                                                        size_x, size_y, size_z)
            x = x1 + x2
            x = F.relu(x)
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.downscale(x)
        x = F.relu(x)
        x = self.output(x)
        
        return x