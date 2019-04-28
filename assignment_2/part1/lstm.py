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

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        self.stdev = 0.001
        normal_dist = Normal(torch.tensor(0.0), torch.tensor(self.stdev))

        # Input-hidden addition
        self.W_xg = nn.Parameter(normal_dist.sample((input_dim, num_hidden)))
        self.W_hg = nn.Parameter(normal_dist.sample((num_hidden, num_hidden)))
        self.b_g = nn.Parameter(normal_dist.sample((1, num_hidden)))

        # Input Gate
        self.W_xi = nn.Parameter(normal_dist.sample((input_dim, num_hidden)))
        self.W_hi = nn.Parameter(normal_dist.sample((num_hidden, num_hidden)))
        self.b_i = nn.Parameter(normal_dist.sample((1, num_hidden)))

        # Forget Gate
        self.W_xf = nn.Parameter(normal_dist.sample((input_dim, num_hidden)))
        self.W_hf = nn.Parameter(normal_dist.sample((num_hidden, num_hidden)))
        self.b_f = nn.Parameter(normal_dist.sample((1, num_hidden)))

        # Output Gate
        self.W_xo = nn.Parameter(normal_dist.sample((input_dim, num_hidden)))
        self.W_ho = nn.Parameter(normal_dist.sample((num_hidden, num_hidden)))
        self.b_o = nn.Parameter(normal_dist.sample((1, num_hidden)))

        # States
        self.h = torch.empty(batch_size, num_hidden, device=device)
        self.c = torch.empty(batch_size, num_hidden, device=device)

        # Hidden Linear Operation
        self.W_hp = nn.Parameter(normal_dist.sample((num_hidden, num_classes)))
        self.b_p = nn.Parameter(normal_dist.sample((1, num_classes)))

    def forward(self, x):
        # Implementation here ...

        # Detach hidden state such that error is not backpropagated into previous sequence
        self.h.detach_()
        self.c.detach_()
        # Reset hidden state
        nn.init.constant_(self.h, 0.0)
        nn.init.constant_(self.c, 0.0)

        for t in range(self.seq_length):
            x_t = x[:, t].view(self.batch_size, self.input_dim)
            g = torch.tanh(x_t @ self.W_xg + self.h @ self.W_hg + self.b_g)
            i = torch.sigmoid(x_t @ self.W_xi + self.h @ self.W_hi + self.b_i)
            f = torch.sigmoid(x_t @ self.W_xf + self.h @ self.W_hf + self.b_f)
            o = torch.sigmoid(x_t @ self.W_xo + self.h @ self.W_ho + self.b_o)
            self.c = g * i + self.c * f
            self.h = torch.tanh(self.c) * o

        p = self.h @ self.W_hp + self.b_p
        return p
