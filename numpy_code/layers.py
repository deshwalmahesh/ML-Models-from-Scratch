import numpy as np


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


class NumpyLoRA:
    def __init__(self, input_feat_dim: int, out_feat_dim: int, rank: int, controlling_alpha: float = 1.0):
        """
        It's just a simple matrix multiplication where matrix "A" gets initialized with random scaled values and "B" gets intialized with zero values
        args:
        input_feat_dim: Number of input features coming from the previous layer
        out_feat_dim: Number of output features that will go out as input to next layer
        rank: determines the "compression" size of LoRA layer and is usually proportional to knowledge gained by LoRA. Increasing it will make the bigger in size (thus more capacity)
        alpha: This is another param I found on the blog:https://lightning.ai/lightning-ai/studios/code-lora-from-scratch#coding-lora-from-scratch. Not sure if it was there in the original
                paper. It basically scales (controls) the results by LoRA. 0.5 means half the effect, 2.0 means double the numbers.
        """
        scaling_factor = 1 / np.sqrt(rank) # in the orig papaer, it scales the random values for Matrix A
        self.A = np.random.randn(input_feat_dim, rank) * scaling_factor  # random weights scaled by the factor. This is input facing matrix (not sure why in the blog was shown as output facing)
        self.B = np.zeros((rank, out_feat_dim))   # output facing matrix. In the beginning it is zero because we want ful pretrained weights contribution as random weights can destabilist the entire system
        self.controlling_alpha = controlling_alpha # not sure if it is needed because it'll scale the "whole" output in "every" direction. Instead we can test it with a learnable param so it adjusts

    def forward(self, input_tensor):
        """
        Performs the forward pass
        args:
            input_tensor: The input tensor to the LoRA layer
        returns: Output of the LoRA layer which will get "ADDED" to the output of the parent layer of this LoRA layer

        """
        return self.controlling_alpha * (input_tensor @ self.A @ self.B)