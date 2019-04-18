"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and
    std = 0.0001. Initialize biases self.params['bias'] with 0.

    Also, initialize gradients with zeros.
    """

    self.grads = {'weight': np.zeros((out_features, in_features)), 'bias': np.zeros(out_features)}
    self.params = {'weight': np.random.normal(0, 0.0001, (out_features, in_features)),
                   'bias': np.zeros(out_features)}

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    self.x = x
    w = self.params['weight']
    b = self.params['bias']
    out = x @ w.T + b

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to
    layer parameters in self.grads['weight'] and self.grads['bias'].
    """

    dx = dout @ self.params['weight']
    self.grads['weight'] = dout.T @ self.x
    self.grads['bias'] = np.sum(dout, axis=0)

    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    self.x = x
    out = np.maximum(0, x)

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    dx = dout * (self.x > 0)

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ex = np.exp(x - np.atleast_2d(np.max(x, axis=1)).T)
    s = np.atleast_2d(np.sum(ex, axis=1)).T
    out = ex / s
    self.out = out

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    n_batch, out_dim = self.out.shape
    diag_tensor = np.zeros((n_batch, out_dim, out_dim))
    diag_idx = np.arange(out_dim)
    diag_tensor[:, diag_idx, diag_idx] = self.out

    within_batch_operation = diag_tensor - np.einsum('ij, ik -> ijk', self.out, self.out)
    dx = np.einsum('ij, ijk -> ik', dout, within_batch_operation)

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    e = 1e-8
    out = np.sum(- y * np.log(x + e), axis=1).mean()
    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """
    e = 1e-8
    dx = - y/(x+e) / y.shape[0]

    return dx
