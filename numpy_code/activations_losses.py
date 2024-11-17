import matplotlib.pyplot as plt
import numpy as np

# ------- Pure NUMPY Based LOSSES -------------
"""
LINK 1: https://aew61.github.io/blog/artificial_neural_networks/1_background/1.b_activation_functions_and_derivatives.html
LINK 2: https://ml-explained.com/blog/activation-functions-explained
LINK 3: https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/activations/activations.py


AMAZING THING: To see the equations of any forward or backward pass, just use the below structure.Install `latexify` package first

@latexify.function(use_math_symbols=True, reduce_assignments=True)
def gelu(X): return 0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X ** 3))) # THIS IS THE __call__ from the GeLU class
gelu # this will render an equation in Notebook
"""

class ElementIndependent():
  def __init__(self):
    self.define = """These are functions which are NOT Dependent on the other elements in an array or Tensor. Ex: Sigmid, ReLU, TanH, Identity aka Linear
    Here Jacobian "𝐉" IS a diagonal matrix and you DON'T have to perform Full Matrix Multiplication"""

  def plot(self, ax = None):
    if ax is None: 
      _, ax = plt.subplots(1,1, figsize = (5,3))

    X = np.linspace(-5,5, 50)
    ax.plot(X, self(X), color = "green", label = "Activation Output")
    ax.plot(X, self.grad(X),  color = "red", label = "Gradients", linestyle = "--")

    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_title(self.__class__.__name__)

    ax.grid()
    ax.legend()
    return ax

class ElementDependent():
  def __init__(self):
    self.define = """These are functions which ARE Dependent on the other elements in an array or Tensor like Softmax
    Here Jacobian "𝐉" is NOT a diagonal matrix and you HAVE TO perform Full Matrix Multiplication"""


class Sigmoid(ElementIndependent):
    def __call__(self, X):
        return 1.0/(1.0 + np.exp(-X))

    def grad(self, X):
        Q = self(X) # calls the __call__()
        return Q*(1-Q)


class Tanh(ElementIndependent):
  def __call__(self, X): return np.tanh(X) # can use (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

  def grad(self, X): return 1.0 - np.tanh(X)**2 # same as 1 - self(X)**2


class SoftPlus(ElementIndependent):
  def __call__(self, X): return np.log(1 + np.exp(X))

  def grad(self, X): return  np.exp(X) / (1 + np.exp(X))


class ReLU(ElementIndependent):
  def __init__(self, alpha = 0):
    """Generalised Relu. Given values of Alpha, it acts as ReLU (alpha = 0), Leaky ReLU (alpha <0), Perametric ReLU (alpha is 'LERANED')"""
    self.alpha = alpha
      
  def __call__(self, X): return np.maximum(self.alpha*X, X) # Can also use np.clip(alpha*X, a_min = 0, a_max = np.inf)

  def grad(self, X):
      """It is NOT differentiable at 0. So people use "Sub Gradients". Check Legendary Convdersation: https://www.quora.com/Why-does-ReLU-work-with-backprops-if-its-non-differentiable"""
      return np.where(X<=0, self.alpha, 1)


class ELU(ElementIndependent):
  def __init__(self, alpha = 1.67326, lmbda = 1.0507, scaled = False):
      "With scaled = True and lmbda != 1, it becomes SeLU"
      self.alpha = alpha
      self.lmbda = lmbda if scaled else 1 # is scaled, multiply by lambda and it becomes SeLU

  def __call__(self, X): return  self.lmbda * (np.where(X < 0, self.alpha*(np.exp(X) - 1), X) )

  def grad(self, X): return self.lmbda * (np.where(X<=0, self.alpha * (np.exp(X)), 1)) # if scaled == 1 it's ELU so return 1 when x> 0 else return according to formula


class SWISH(ElementIndependent):
  def __call__(self, X):
      """Unlike ReLU, it's smooth at X == 0"""
      return X / (1 + np.exp(-X)) # can use X * Sigmoid()(X) too

  def grad(self, X):
      forward_X = self(X)
      return forward_X + (Sigmoid()(X) * (1-forward_X))

class GeLU(ElementIndependent):
  def __call__(self, X):
    """Taken From: https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/activations/activations.py"""
    # We're using the "approximation" using Tanh. You can also use the Sigmoid and CDF based one too: https://cdn.prod.website-files.com/5d7b77b063a9066d83e1209c/60be260e5e9ac0bc4d8beac9_math-20210607%20(17).png
    return 0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X ** 3))) 
  
  def grad(self, X):
     s = X / np.sqrt(2)
     erf_prime = lambda x: (2 / np.sqrt(np.pi)) * np.exp(-(X ** 2))
     approx = np.tanh(np.sqrt(2 / np.pi) * (X + 0.044715 * X ** 3))
     return 0.5 + 0.5 * approx + ((0.5 * X * erf_prime(s)) / np.sqrt(2))

class MISH(ElementIndependent):
  def __init__(self): raise NotImplementedError()
