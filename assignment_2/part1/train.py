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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    if config.device == 'cuda:0':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.device = device
    else:
        device = torch.device('cpu')
        config.device = device

    # Initialize the model that we are going to use
    model = None
    if config.model_type == 'RNN':
        model = VanillaRNN(seq_length=config.input_length,
                           input_dim=config.input_dim,
                           num_hidden=config.num_hidden,
                           num_classes=config.num_classes,
                           batch_size=config.batch_size,
                           device=device)
    elif config.model_type == 'LSTM':
        model = LSTM(seq_length=config.input_length,
                     input_dim=config.input_dim,
                     num_hidden=config.num_hidden,
                     num_classes=config.num_classes,
                     batch_size=config.batch_size,
                     device=device)
    else:
        print("You have no model, bro!")
        quit()
    model = model.to(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=0)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    accuracy_train = []
    loss_train = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...
        x = batch_inputs.to(device)
        y = batch_targets.to(device)

        # Forward pass
        p = model.forward(x)
        loss = criterion(p, y)
        loss_train.append(loss)
        optimizer.zero_grad()
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        # ANSWER: This clips the gradient to the given value. This prevents the
        # gradient growing exponentially, preventing the gradient exploding problem.
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        # Add more code here ...
        optimizer.step()

        accuracy = torch.sum(torch.argmax(p, dim=1) == y).to(torch.float) / float(config.batch_size)
        accuracy_train.append(accuracy)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 100 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.train_steps or np.mean(accuracy_train[-100:]) == 1.0:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    if np.mean(accuracy_train[-100:]) == 1.0:
        mean = np.mean(accuracy_train[-100:])
    else:
        mean = np.mean(accuracy_train[np.argmax(accuracy_train)-50:np.argmax(accuracy_train)+50])
        print(np.max(accuracy_train))
    return np.max(accuracy_train), mean, len(accuracy_train)


def run_experiment(config):
    start = 10
    end = 35
    import matplotlib.pyplot as plt
    results = []
    for i in range(start, end+1):
        print("Sentence length:", i)
        config.input_length = i
        data = train(config)
        results.append(data[1])
        results_array = np.array(results)

        if i != start:
            plt.plot(np.arange(start, i+1), results_array)
            plt.ylabel("Accuracy")
            plt.xlabel("Sentence length")
            plt.title(config.model_type +" maximum running mean accuracy (100 samples) \n with varying sentence lengths")
            plt.savefig("poop_" + config.model_type + ".pdf")

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    # train(config)
    run_experiment(config)