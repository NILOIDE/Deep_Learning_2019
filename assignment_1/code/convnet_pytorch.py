"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """
    class Flatten(nn.Module):
      def forward(self, x):
        return x.view(x.size(0), -1)

    super(ConvNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes

    self.layer_specs = [{'name': 'conv1',
                         'kernel_size': 3, 'stride': 1, 'padding': 1, 'in_channels': n_channels, 'out_channels': 64},
                        {'name': 'maxpool1',
                         'kernel_size': 3, 'stride': 2, 'padding': 1, 'in_channels': 64, 'out_channels': 64},
                        {'name': 'conv2',
                         'kernel_size': 3, 'stride': 1, 'padding': 1, 'in_channels': 64, 'out_channels': 128},
                        {'name': 'maxpool2',
                         'kernel_size': 3, 'stride': 2, 'padding': 1, 'in_channels': 128, 'out_channels': 128},
                        {'name': 'conv3_a',
                         'kernel_size': 3, 'stride': 1, 'padding': 1, 'in_channels': 128, 'out_channels': 256},
                        {'name': 'conv3_b',
                         'kernel_size': 3, 'stride': 1, 'padding': 1, 'in_channels': 256, 'out_channels': 256},
                        {'name': 'maxpool3',
                         'kernel_size': 3, 'stride': 2, 'padding': 1, 'in_channels': 256, 'out_channels': 256},
                        {'name': 'conv4_a',
                         'kernel_size': 3, 'stride': 1, 'padding': 1, 'in_channels': 256, 'out_channels': 512},
                        {'name': 'conv4_b',
                         'kernel_size': 3, 'stride': 1, 'padding': 1, 'in_channels': 512, 'out_channels': 512},
                        {'name': 'maxpool4',
                         'kernel_size': 3, 'stride': 2, 'padding': 1, 'in_channels': 512, 'out_channels': 512},
                        {'name': 'conv5_a',
                         'kernel_size': 3, 'stride': 1, 'padding': 1, 'in_channels': 512, 'out_channels': 512},
                        {'name': 'conv5_b',
                         'kernel_size': 3, 'stride': 1, 'padding': 1, 'in_channels': 512, 'out_channels': 512},
                        {'name': 'maxpool5',
                         'kernel_size': 3, 'stride': 2, 'padding': 1, 'in_channels': 512, 'out_channels': 512},
                        {'name': 'avgpool',
                         'kernel_size': 1, 'stride': 1, 'padding': 0, 'in_channels': 512, 'out_channels': 512},
                        {'name': 'linear', 'in_features': 512, 'out_features': n_classes}
                        ]
    self.layers = []
    for layer in self.layer_specs:
      if layer['name'][:4] == 'conv':
        args = {p: layer[p] for p in layer if p != 'name'}
        self.layers.append(nn.Conv2d(**args))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm2d(layer['out_channels']))
      elif layer['name'][:7] == 'maxpool':
        args = {p: layer[p] for p in layer if p != 'name' and p != 'in_channels' and p != 'out_channels'}
        self.layers.append(nn.MaxPool2d(**args))
      elif layer['name'][:7] == 'avgpool':
        args = {p: layer[p] for p in layer if p != 'name' and p != 'in_channels' and p != 'out_channels'}
        self.layers.append(nn.AvgPool2d(**args))
      elif layer['name'][:6] == 'linear':
        args = {p: layer[p] for p in layer if p != 'name'}
        self.layers.append(Flatten())
        self.layers.append(nn.Linear(**args))
    self.model = nn.Sequential(*self.layers)

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    out = self.model(x)

    return out
