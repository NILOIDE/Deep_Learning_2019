"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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

  accuracy = (np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)).mean()

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Load data set --------------------
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  test_images, y_test = cifar10['test'].images, cifar10['test'].labels
  test_img_num, im_channels, im_height, im_width = test_images.shape
  x_size = im_channels * im_height * im_width
  x_test = test_images.reshape((test_img_num, x_size))
  # ----------------------------------
  # Create MLP -----------------------
  mlp = MLP(x_size, dnn_hidden_units, y_test.shape[1])
  CE_module = CrossEntropyModule()
  # ----------------------------------

  results = []
  batch_size = FLAGS.batch_size
  for epoch in range(1, FLAGS.max_steps+1):
    # Prepare batch -------------------------
    x_train, y_train = cifar10['train'].next_batch(batch_size)
    x_train = x_train.reshape((batch_size, x_size))
    # ---------------------------------------
    # Train step ----------------------------
    output = mlp.forward(x_train)
    train_loss = CE_module.forward(output, y_train)
    dCE = CE_module.backward(output, y_train)
    mlp.backward(dCE)
    for layer in mlp.param_layers:
      layer.params['weight'] -= FLAGS.learning_rate * layer.grads['weight']
      layer.params['bias'] -= FLAGS.learning_rate * layer.grads['bias']
    # ----------------------------------------
    # Store every eval_freq steps ------------
    if epoch % FLAGS.eval_freq == 0:
      train_acc = accuracy(output, y_train)
      test_output = mlp.forward(x_test)
      test_loss = CE_module.forward(test_output, y_test)
      test_acc = accuracy(test_output, y_test)
      results.append({'Train loss': train_loss, 'Train accuracy': train_acc,
                      'Test loss': test_loss, 'Test accuracy': test_acc})
      print("Epoch:", epoch, "  Loss:", train_loss, "Acc:", train_acc)
    # ----------------------------------------

  if results:
    import matplotlib.pyplot as plt
    y_axis = {'Train loss': [r['Train loss'] for r in results],
              'Train accuracy': [r['Train accuracy'] for r in results],
              'Test loss': [r['Test loss'] for r in results],
              'Test accuracy': [r['Test accuracy'] for r in results]}
    x_axis = np.arange(len(results))*FLAGS.eval_freq
    plt.plot(x_axis, y_axis['Train loss'], x_axis, y_axis['Train accuracy'],
             x_axis, y_axis['Test loss'], x_axis, y_axis['Test accuracy'])
    plt.legend(['Train loss', 'Train accuracy', 'Test loss', 'Test accuracy'])
    plt.xlabel("Training steps")
    plt.ylabel("Accuracy / Loss")
    plt.savefig("mlp_numpy_curves.pdf")

    print("--------Best Results--------")
    best_idx = np.argmax(y_axis['Test accuracy'])
    print("Best epoch:", best_idx*FLAGS.eval_freq)
    best_idx = np.argmax(y_axis['Test accuracy'])
    for s, r in zip([*y_axis], [y_axis[i][best_idx] for i in y_axis]):
      print(s, r)
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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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
