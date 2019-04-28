################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        self.stdev = 0.01

        self.W_xh = nn.Parameter(Normal(torch.tensor(0.0), torch.tensor(self.stdev)).sample((input_dim, num_hidden)))
        self.W_hh = nn.Parameter(Normal(torch.tensor(0.0), torch.tensor(self.stdev)).sample((num_hidden, num_hidden)))
        self.W_hp = nn.Parameter(Normal(torch.tensor(0.0), torch.tensor(self.stdev)).sample((num_hidden, num_classes)))

        self.b_h = nn.Parameter(Normal(torch.tensor(0.0), torch.tensor(self.stdev)).sample((1, num_hidden)))
        self.b_p = nn.Parameter(Normal(torch.tensor(0.0), torch.tensor(self.stdev)).sample((1, num_classes)))

        self.h = torch.empty(self.batch_size, self.num_hidden, device=self.device)

    def forward(self, x):
        # Implementation here ...

        # Detach hidden state such that error is not backpropagated into previous sequence
        self.h.detach_()
        # Reset hidden state
        self.h = torch.zeros(self.batch_size, self.num_hidden, device=self.device)

        for t in range(self.seq_length):
            x_t = x[:, t].view(self.batch_size, self.input_dim)
            self.h = torch.tanh(x_t @ self.W_xh + self.h @ self.W_hh + self.b_h)

        out = self.h @ self.W_hp + self.b_p

        return out
