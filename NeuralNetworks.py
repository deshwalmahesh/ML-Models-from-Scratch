import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data



nnfs.init()

X, y = spiral_data(100,3)


class DenseLayer:
    '''
    Class to Implement the Fully Connected or the Dense Layer
    '''
    def __init__(self,input_size:int,output_size:int)->None:
        '''
        args:
            input_size: Size of the data coming from previous layer or the No of features coming from the Input Layer directly
            output_size: Number of Neurons or the size of the output of this layer
        '''
        self.input_size = input_size
        self.output_size  = output_size
        self.W = np.random.randn(self.input_size, self.output_size,) # Weight matrix corresponding to each neuron. To avoid Transposing the Weight Matrix, we used it as (input_size, output_size,) else it would have been the opposite
        self.b = np.zeros((1, self.output_size)) # We have to add one bias for each neuron
    
    
    def forward(self,X:np.ndarray)->None:
        '''
        Make a forward pass based on the equation y = W.x + b and store the results
        args:
            Pass in the result as numpy array of features input or output from another layer
        '''
        self.output = np.dot(X, self.W) + self.b # y = W.X + b  (X dot (W_Transpose)) 


class ReLuActivation:
    '''
    Class to implement the Rectified Linear Unit as Activation Function
    '''
    def forward(self,X:np.ndarray)->None:
        '''
        Give the results coming from any Layers as a maximum of 0 or the input for each row
        '''
        self.output = np.maximum(0, X)


