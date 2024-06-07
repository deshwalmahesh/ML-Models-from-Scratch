"""
Loss, Layers, Activations
"""

import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Dense:
    '''
    Class to Implement the Fully Connected or the Dense Layer
    '''
    def __init__(self,input_size:int,output_size:int,factor:float=0.01)->None:
        '''
        args:
            input_size: Size of the data coming from previous layer or the No of features coming from the Input Layer directly
            output_size: Number of Neurons or the size of the output of this layer
            factor: Normalization factor for initial weights 
        '''
        self.input_size = input_size
        self.output_size  = output_size
        self.W = factor * np.random.randn(self.input_size, self.output_size,) # Weight matrix corresponding to each neuron. To avoid Transposing the Weight Matrix, we used it as (input_size, output_size,) else it would have been the opposite
        self.b = np.zeros((1, self.output_size)) # We have to add one bias for each neuron
    
    
    def forward(self,X:np.ndarray)->None:
        '''
        Make a forward pass based on the equation y = W.x + b and store the results
        args:
            Pass in the result as numpy array of features input or output from another layer
        '''
        self.output = np.matmul(X, self.W) + self.b # y = W.X + b  (X dot (W_Transpose)) 


class ReLu:
    '''
    Class to implement the Rectified Linear Unit as Activation Function
    '''
    def forward(self,X:np.ndarray)->None:
        '''
        Give the results coming from any Layers as a maximum of 0 or the input for each row
        args:
            X: Input data of shape (batch_size, No of features)
        '''
        self.output = np.maximum(0, X)


class Softmax:
    '''
    Apply Softmax Activation Function
    '''
    def forward(self,X:np.ndarray)->None:
        '''
        Forward pass to return the normalized values from the input values
        args:
            X: Input data of shape (batch_size, No of features)
        '''
        # axis=1, keepdims=True parameter is used as we work with batches not single input so the correct value is operated from itself 
        max_value_per_instance = np.max(X, axis=1, keepdims=True) # Get max for each input and Center the Values so that you don't blow the exponential as Max will be 0 and others will be -ive
        centered_exp = np.exp(X - max_value_per_instance) # Get the exponential values for each data point
        self.output = centered_exp / np.sum(centered_exp, axis=1, keepdims=True) # apply the formula as e^x / sum(e^x) for each input in the batch 


class Loss:
    '''
    A Generic Loss class that'll act as a parent or Base class for classes which need to accumate or calculate the loss
    '''
    def average_loss(self,y_pred:np.ndarray,y_true:np.ndarray)->float:
        '''
        Accumulate the Loss. The Loss will be calculated by each of it's Child Class. The code to calculate the Loss given from y_true and y_pred will be dependent on type of loss so that is
        why we have made this a base class. Check the next Class for how it'll be used.
        args:
            y_pred: Predicted values
            y_true: Original values
        out:
            Returns the Mean Loss based on all the samples
        '''
        return np.mean(self.forward(y_pred, y_true)) # Return the average loss of a particular batch data. forward() function will be implemented in Child class because loss calculation is task specific


class CategoricalCrossEntropy(Loss):
    '''
    Class to calculate the categorical cross entropy (or log loss). It inherits the Loss() as its parent class which accumulates the loss
    '''
    def forward(self,y_pred:np.ndarray,y_true:np.ndarray)->float:
        '''
        Make a forward pass to the batch of data to calculate the loss. This method is used in Loss() class for accumulation of loss inside average_loss()
        args:
            y_pred: Predicted values. This has to be in One-Hot encoded form
            y_true: Ground Truth Values. These has to be One - Hot encoded too
        out:
            Returns the Log loss for each input data
        '''
        n = y_pred.shape[0] # No of samples or batch size
        clipped = np.clip(y_pred, 1e-7,1-1e-7) # Keep the values within range 1*10^-7 and 0.99999999 (because log loss can never be exactly 0 or 1). Doing this helps in preventing Shifting the Average Loss value due to any Big or Small value
        if len(y_true.shape) == 1: # If y_true are "Sparse" and there are 6 Samples in batch then y_true would be [0,1,0,2,1,2] but y pred would be of shape [Batch Size, No of Neurons]
            result = clipped[range(n), y_true] # It'll give the neuron value for the corresponding index. It'll give the value of 0th neuron, 1st neuron, 0th neuron, 2nd neuron and so on for each of 6 samples

        elif len(y_true.shape) == 2: # If Y true are one hot encodes. So BOTH y_true and y_pred will have shape [No of samples, No of classes]. y_pred = [[0.2,0.5,0.3],..] and y_true = [[0,1,0],...] 
            result = np.sum(clipped * y_true, axis = 1) # This is the part where we multiply the Exact value of the corresponding Neuron. Other neurons will be Null as the  [[0.2*0, 0.5*1, 0.3*0],..]

        return -np.log(result) # This is Log Likelihood of Cross Entropy of a distribution. We'll have to take the average of this in the Loss() class     


class EvaluationMetrics:
    '''
    Different Metrics depending on different use cases for the evaluation of model
    '''
    def accuracy(self,y_true:np.ndarray,y_pred:np.ndarray)->float:
        '''
        Get the accuracy from predicted vs Actual class. Tells how many items were actually correctly classified
        args:
            y_true: 1  (Sparse) or 2-D (One hot) Array of Actual class
            y_pred: 1  1 or 2-D (One hot) Array of predicted class
        out: 
            returns the portion of classes correctly identified in the data 0.8 means 80%
        '''
        if len(y_true.shape) == 2: # one hot
            y_true = np.argmax(y_true,axis=1) # It'll be in form of [[0,1,0], [1,0,0]] etc given three classes
        
        if len(y_pred.shape) == 2: # one hot values for more than 1 class given in Categorical Cross Entropy
            y_pred = np.argmax(y_pred,axis=1) # [[0.1,0.2,0.7], [0.8,0.1,0.1]]
            
        return np.mean(y_true == y_pred)


# ------- Pure NUMPY -------------
"""
LINK: https://aew61.github.io/blog/artificial_neural_networks/1_background/1.b_activation_functions_and_derivatives.html
"""

class ElementIndependent():
  def __init__(self):
    self.define = """These are functions which are NOT Dependent on the other elements in an array or Tensor. Ex: Sigmid, ReLU, TanH, Identity aka Linear
    Here Jacobian "𝐉" IS a diagonal matrix and you DON'T have to perform Full Matrix Multiplication"""

  def plot(self, ax = None):
    if ax is None: _, ax = plt.subplots(1,1, figsize = (5,3))

    X = np.linspace(-5,5, 50)
    plt.suptitle(self.__class__.__name__, size = 15)
    ax.plot(X, self(X), color = "green", label = "Activation Output")
    ax.plot(X, self.grad(X),  color = "red", label = "Gradients")

    ax.set_xlabel("Input")
    ax.set_ylabel("Output")

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
  def __call__(self, X):
      return np.tanh(X) # can use (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

  def grad(self, X):
      """Direct Numpy"""
      return 1.0 - np.tanh(X)**2 # same as 1 - self(X)**2


class ReLU(ElementIndependent):
  def __init__(self, alpha = 0, is_elu = False):
    """Generalised Relu. Given values of Alpha(Slope) and is_elu, it acts as ReLU (alpha = 0), Leaky ReLU (alpha <0), Perametric ReLU (alpha is 'LERANED') and ELU"""
    self.alpha = alpha
    self.is_elu = is_elu
    self.__class__.__name__ = self.__class__.__name__ + f" (Alpha = {alpha} + ELU = {is_elu})"

  def __call__(self, X):
    if not self.is_elu: return np.maximum(self.alpha*X, X) # Can also use np.clip(alpha*X, a_min = 0, a_max = np.inf)
    else: return np.where(X < 0, 0.1*(np.exp(X) - 1), X)

  def grad(self, X):
      """
      It is NOT differentiable at 0. So people use "Sub Gradients". Check Legendary Convdersation: https://www.quora.com/Why-does-ReLU-work-with-backprops-if-its-non-differentiable
      """
      if not self.is_elu: return np.where(X<=0, self.alpha, 1)
      return np.where(X<=0, self.alpha*(np.exp(X)), 1)
