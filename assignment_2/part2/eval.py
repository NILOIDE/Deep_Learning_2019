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


def one_hot(batch, vocab_size):
    one_hot_batch = torch.zeros((len(batch), *batch[0].shape, vocab_size))
    return one_hot_batch.scatter_(2, batch.unsqueeze(-1), 1)


def generate(sentences, model, seed, dataset, config, print_nl=False):
    # Propagate seed
    seed_one_hot = one_hot(sentences[:, :seed.shape[0]], dataset.vocab_size)
    p, last_state = model.forward(seed_one_hot, None)
    distribution = torch.softmax(p[:,-1,:].squeeze(1) / config.temperature, dim=1)
    char = torch.multinomial(distribution, 1)
    sentences[:, seed.shape[0]] = char.squeeze(1)
    last_state = last_state
    for l in range(seed.shape[0], sentences.shape[1]):
        x = one_hot(char.clone().detach(), dataset.vocab_size)
        p, last_state = model.forward(x, last_state)
        distribution = torch.softmax(p.squeeze(1) / config.temperature, dim=1)
        char = torch.multinomial(distribution, 1)
        sentences[:, l] = char.squeeze(1)
    if print_nl:
        text = [dataset.convert_to_string(sentence.tolist()) for sentence in sentences]
    else:
        text = [dataset.convert_to_string(sentence.tolist()).replace('\n', '\\n ') for sentence in sentences]
    return text


def load_model():
    while True:
        model_name = input("Input model name:")
        if os.path.exists(model_name):
            model = torch.load(model_name, map_location='cpu')
            steps = int(np.load(model_name[:-3] + "_elapsed.npy"))
            return model, steps
        else:
            print("Model not found!")


def eval():

    # torch.set_default_tensor_type(torch.LongTensor)

    # Initialize the dataset
    dataset = TextDataset(config.txt_file, config.sample_len)

    model, steps = load_model()
    print("Model trained for", steps, "steps")
    model.eval()

    while True:
        # Get input for the start of the sentence
        seed = input("\nStart: ")
        print()
        # Convert input to one-hot representation (length x vocab_size)
        try:
            sentences = torch.zeros((config.num_samples, config.sample_len + len(seed)), dtype=torch.long)
            seed = torch.tensor([dataset._char_to_ix[ch] for ch in seed])
            sentences[:, :len(seed)] = seed
            generated_samples = generate(sentences, model, seed, dataset, config)

            paragraph = torch.zeros((1, config.paragraph_len + len(seed)), dtype=torch.long)
            paragraph[:, :len(seed)] = seed
            generated_paragraph = generate(paragraph, model, seed, dataset, config, print_nl=True)

        except KeyError:
            print("One or more characters were not recognized. Try again!")
            continue

        print("Sentences:\n")
        for s in generated_samples:
            print("-", s, "\n")
        print("\nParagraph:\n")
        for p in generated_paragraph:
            print(p)


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse evaluation configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='Viimeinen_syksy_clean.txt', help="Path to a .txt file to train on")
    parser.add_argument('--sample_len', type=int, default=100, help='Length of an input sequence')
    parser.add_argument('--paragraph_len', type=int, default=1000, help='Length of an input sequence')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of examples to process in a batch')
    parser.add_argument('--temperature', type=float, default=0.25, help='Temperature when sampling the next character')

    config = parser.parse_args()

    # Evaluate the model
    eval()
