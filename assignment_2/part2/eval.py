from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

from tensorboardX import SummaryWriter

################################################################################


def get_one_hot(start, dataset):
    one_hot = torch.zeros(len(start), dataset.vocab_size)
    for idx, c in enumerate(start):
        one_hot[idx] = torch.tensor(dataset.convert_to_one_hot(c))
    return one_hot


def load_model(model):
    if os.path.exists('model.tar'):
        checkpoint = torch.load('model.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        steps = checkpoint['step']
    else:
        print("Model not found!")
        quit()
    return model, steps


def eval():
    # Torch settings
    torch.set_default_tensor_type(torch.FloatTensor)

    # Initialize the dataset
    dataset = TextDataset(config.txt_file, config.seq_length)

    # Get temperature
    temp = config.temperature

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size,
                                config.seq_length,
                                dataset.vocab_size)

    # Load model, if there's any model to load
    model, steps = load_model(model)
    print("Model trained for", steps, "steps")
    model.eval()

    try:
        while True:
            # Get input for the start of the sentence
            start = input("\nStart: ")

            # Convert input to one-hot representation (length x vocab_size)
            try:
                start_oh = get_one_hot(start, dataset)
            except KeyError:
                print("One or more characters were not recognized. Try again!")
                continue

            # Generate the rest of the sentence
            sentence = dataset.convert_to_string(model.cmd_generate(start_oh, temp, config.seq_length))

            print("Model says:\n")
            print(start + sentence)
    except KeyboardInterrupt:
        print("\n\nPromise you'll call me back!")


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse evaluation configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='data/dumas.txt', help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature when sampling the next character')

    config = parser.parse_args()

    # Evaluate the model
    eval()
