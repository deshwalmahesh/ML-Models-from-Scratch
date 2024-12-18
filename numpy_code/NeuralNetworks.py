"""
Loss, Activations
"""

import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

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
