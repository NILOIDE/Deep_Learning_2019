"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  accuracy = torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1)
  accuracy = accuracy.float().mean()

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)

  # Pytorch stuff --------------------
  data_type = torch.FloatTensor
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # Load data set --------------------
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  x_test, y_test = cifar10['test'].images, cifar10['test'].labels
  test_img_num, im_channels, im_height, im_width = x_test.shape
  # ----------------------------------
  # Create MLP -----------------------
  model = ConvNet(im_channels, y_test.shape[1])
  model = model.to(device)
  CE_module = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
  # ----------------------------------

  train_results = []
  test_results =[]
  for epoch in range(1, FLAGS.max_steps+1):
    # Prepare batch -------------------------
    x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
    x_train = torch.tensor(x_train).type(data_type).to(device)
    y_train = torch.tensor(y_train).type(data_type).to(device)
    # ---------------------------------------
    # Train step ----------------------------
    optimizer.zero_grad()
    output = model.forward(x_train)
    train_loss = CE_module.forward(output, torch.argmax(y_train, dim=1))
    train_loss.backward()
    optimizer.step()
    # ----------------------------------------
    # Store every eval_freq steps ------------
    train_acc = accuracy(output.detach(), y_train)
    train_results.append([epoch, train_loss.detach().item(), train_acc.item()])

    if epoch % FLAGS.eval_freq == 0 or epoch == 1:
      t_size = FLAGS.batch_size
      test_loss = []
      test_output = None
      test_labels = None
      for i in range(t_size, test_img_num, t_size):
        x_test_batch, y_test_batch = cifar10['test'].next_batch(t_size)
        x_test_batch = torch.tensor(x_test_batch).type(data_type).to(device)
        y_test_batch = torch.tensor(y_test_batch).type(data_type).to(device)
        test_batch_output = model.forward(x_test_batch)
        test_batch_loss = CE_module.forward(test_batch_output, torch.argmax(y_test_batch, dim=1))
        test_loss.append(test_batch_loss.detach().item())
        if test_output is None:
          test_output = test_batch_output.detach()
        else:
          test_output = torch.cat((test_output, test_batch_output.detach()), 0)
        if test_labels is None:
          test_labels = test_batch_output.detach()
        else:
          test_labels = torch.cat((test_labels, y_test_batch.detach()), 0)
      test_acc = accuracy(test_output, test_labels)
      test_results.append([epoch, np.sum(test_loss)/(test_img_num/t_size), test_acc.item()])
    # ----------------------------------------

  if train_results and test_results:
    import matplotlib.pyplot as plt
    train_results = np.array(train_results)
    train_x_axis = train_results[:, 0]
    train_loss = train_results[:, 1]
    train_acc = train_results[:, 2]
    test_results = np.array(test_results)
    test_x_axis = test_results[:, 0]
    test_loss = test_results[:, 1]
    test_acc = test_results[:, 2]
    plt.plot(train_x_axis, train_loss, train_x_axis, train_acc,
             test_x_axis, test_loss, test_x_axis, test_acc)
    plt.legend(['Train loss', 'Train accuracy', 'Test loss', 'Test accuracy'])
    plt.xlabel("Training steps")
    plt.ylabel("Accuracy / Loss")
    plt.savefig("convnet_pytorch_curves.pdf")

    print("--------Best Results--------")
    best_idx = np.argmax(test_results[:, 2])
    print("Best epoch:", best_idx*FLAGS.eval_freq)
    print("Train loss", train_results[best_idx, 1])
    print("Train accuracy", train_results[best_idx, 2])
    print("Test loss", test_results[best_idx, 1])
    print("Test accuracy", test_results[best_idx, 2])
    print("-----------------------------")

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()