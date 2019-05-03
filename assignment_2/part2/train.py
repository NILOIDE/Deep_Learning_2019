# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def one_hot(batch, vocab_size):
    one_hot_batch = torch.zeros((len(batch), *batch[0].shape, vocab_size)).to(batch.device)
    return one_hot_batch.scatter_(2, batch.unsqueeze(-1), 1)


def sample(p, t):
    distribution = torch.softmax(p / t, dim=0)
    return torch.multinomial(distribution, 1)


def generate(model, dataset, config):
    with torch.no_grad():
        sentences = torch.zeros((1, config.sample_len)).to(config.device)
        # Give first letter of samples as seed
        char = torch.randint(low=0, high=dataset.vocab_size, size=(1, 1))
        sentences[:, 0] = char.squeeze(1)
        last_state = None
        for l in range(1, config.sample_len):
            x = one_hot(torch.tensor(char.detach().clone(), dtype=torch.long), dataset.vocab_size).to(config.device)

            # sample next letter for all sentences
            p, last_state = model(x, last_state)
            char = sample(p.squeeze(1), config.temperature)
            sentences[:, l] = char.squeeze(1)

        # Convert from vocab index to character
        text = [dataset.convert_to_string(sentence.tolist()).replace('\n', '\\n ') for sentence in sentences]
    return text

# def generate(model, dataset, config):
#     temperature = config.temperature
#     device = config.device
#     length = config.sample_len
#     seed = torch.randint(low=0, high=dataset.vocab_size, size=(1, 1))
#     with torch.no_grad():
#         # the seed is the very first leter of the text
#         text = seed.view(-1).tolist()
#
#         # convert seed into a one-hot representation
#         seed = one_hot(seed, dataset.vocab_size)
#
#         # forward pass through the model
#         y, (h, c) = model(seed)
#
#         # sample a character from the output
#         next_char = sample(y[:, -1, :], temperature)
#
#         text.append(next_char.item())
#
#         # can this loop be prettier?
#         for l in range(length - 1):
#             # one-hot encode the previous carachter
#             x = one_hot(torch.tensor(next_char,
#                                      dtype=torch.long,
#                                      device=device).view(1, -1),
#                         dataset.vocab_size)
#
#             # forward pass through the model
#             y, (h, c) = model(x, (h, c))
#
#             # sample a character from the output
#             next_char = sample(y[:, -1, :], temperature)
#
#             # append the char to the text
#             text.append(next_char.item())
#
#         # convert indexes into chars
#         text = dataset.convert_to_string(text)
#
#     return text


def train(config):

    # Initialize the device which to run the model on
    if config.device == 'cuda:0':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.device = device
    else:
        device = torch.device('cpu')
        config.device = device

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)

    data_loader = DataLoader(dataset, config.batch_size, num_workers=0)

    if config.load_name is None:
        # Initialize the model that we are going to use
        model = TextGenerationModel(batch_size=config.batch_size,
                                    seq_length=config.seq_length,
                                    vocabulary_size=dataset.vocab_size,
                                    lstm_num_hidden=config.lstm_num_hidden,
                                    lstm_num_layers=config.lstm_num_layers,
                                    dropout=(1-config.dropout_keep_prob),
                                    device=device)
        accuracy_train = []
        steps_elapsed = 0
    else:
        model = torch.load(config.load_name, map_location=device)
        accuracy_train = list(np.load(config.load_name[:-3] + "_accuracy.npy"))
        steps_elapsed = int(np.load(config.load_name[:-3] + "_elapsed.npy"))

    model = model.to(device)
    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate)

    while steps_elapsed < config.train_steps:
        for step, (batch_inputs, batch_targets) in enumerate(data_loader, start=steps_elapsed):

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################
            batch_inputs = torch.stack(batch_inputs, dim=1).to(device)
            x = one_hot(batch_inputs, dataset.vocab_size)

            y = torch.stack(batch_targets, dim=1).to(device)
            # y = one_hot(batch_targets, dataset.vocab_size).to(device)

            p, lstm_state = model.forward(x)
            loss = criterion(p.transpose(2, 1), y)

            optimizer.zero_grad()
            loss.backward()
            accuracy = torch.sum(torch.argmax(p, dim=2) == y).to(torch.float32) / (config.batch_size * config.seq_length)
            accuracy_train.append(accuracy.item())

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)
            steps_elapsed += 1

            if steps_elapsed % config.learning_rate_step == 0:
                config.learning_rate *= config.learning_rate_decay
                for group in optimizer.param_groups:
                    group['lr'] = config.learning_rate

            if steps_elapsed % config.print_every == 0:
                # print(f"Train Step {step+1}/{config.train_steps}, Examples/Sec = {examples_per_second},"
                #       f" Accuracy = {accuracy}, Loss = {loss}")
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), int(step+1),
                        int(config.train_steps), config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if steps_elapsed % config.sample_every == 1 or step == 0:
                model.eval()
                test_temperature = [0.0001, 0.25, 0.5, 1.0, 2.0]
                og_sample_len = config.sample_len
                for t in test_temperature:
                    config.temperature = t
                    config.sample_len = og_sample_len
                    generated_samples = generate(model, dataset, config)
                    print("-------------------------------------------")
                    print("Temperature: " + str(t))
                    print("Generated " + str(config.num_samples) + ":")
                    for s in generated_samples:
                        print(s)
                    config.sample_len = 100
                    generated_samples = generate(model, dataset, config)
                    print("\nGenerated " + str(config.num_samples) + " long samples (" + str(config.sample_len) + " chars):")
                    for s in generated_samples:
                        print(s)
                print("-------------------------------------------")
                config.sample_len = og_sample_len
                model.train()

            if steps_elapsed % config.save_every == 0 or step == 0:
                # Save the final model
                file_name = config.txt_file[:-4] + "_" + str(steps_elapsed) + "_RMS_opt_model"
                torch.save(model, file_name + ".pt")
                np.save(file_name + "_accuracy", accuracy_train)
                np.save(file_name + "_elapsed", steps_elapsed)
                print("Saved model.")

            if steps_elapsed == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

    print('Done training.')
    print("**************************************************************")


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')


    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=1000, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=50000, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--num_samples', type=int, default=5, help='How many sentences to sample')
    parser.add_argument('--sample_len', type=int, default=30, help='How long sampled sentences are')
    parser.add_argument('--save_every', type=int, default=50000, help='How often to save the model')
    parser.add_argument('--load_name', type=str, default=None, help='Which model to load')

    config = parser.parse_args()

    # Train the model
    train(config)
