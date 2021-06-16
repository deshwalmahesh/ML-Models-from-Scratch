import numpy as np
import math


class Activations:
    '''
    Class to define the activations used in LSTM Network. Tanh and Sigmoid
    '''

    def sigmoid(self,x:np.ndarray)->np.ndarray:
        '''
        Sigmoid activation function
        args:
            x: Batch of incoming vectors
        '''
        return 1/(1+math.exp(-x))

    def tanh(self,x:np.ndarray)->np.ndarray:
        '''
        Hyperbolic Tangent Activation Function
        args:
            x: Batch of incoming vectors
        '''
        return np.tanh(x)

    def derivative_sigmoid(self,x:np.ndarray)->np.ndarray:
        '''
        d_sigmoid(x) / d_x. Used in Backward Pass
        args:
            x: Batch of incoming vectors
        '''
        return x * (1 - x)
    
    def derivative_tanh(self,x:np.ndarray)->np.ndarray:
        '''
        d_tanh(x) / d_x Used in Backward Pass
        args:
            x: Batch of incoming vectors
        '''
        return 1 - x * x


class LSTM:
    '''
    LSTM Layer
    '''
    def __init__(self,H_size:int, X_size:int):
        '''
        args:
            H_size: Hidden Size Vector. All the hidden weight matrics and vectors will be of this size
            X_size: Dimension of input vector. Depends on the Embedding size of each input. Ex: Glove have 100,200,300 size so each word will have any of these sizes
        '''
        z_size = H_size + X_size # Size of concat vector
        self.sigmoid = Activations().sigmoid
        self.tanh = Activations().tanh

        self.W_f = np.random.randn(H_size, z_size) # Weight for Forget Gate Matrix
        self.b_f = np.zeros((H_size, 1)) # Bias for above

        self.W_i = np.random.randn(H_size, z_size)  # Weight for Intermediate gate Matrix
        self.b_i = np.zeros((H_size, 1))

        self.W_c = np.random.randn(H_size, z_size) # Weight for Cell State Gate
        self.b_c = np.zeros((H_size, 1))

        self.W_o = np.random.randn(H_size, z_size) # Matrix weight for output gate which will produce the Current Hidden State
        self.b_o = np.zeros((H_size, 1))

       
        self.W_v = np.random.randn(X_size, H_size)  # For final layer to predict the next character
        self.b_v = np.zeros((X_size, 1))

    
    def forward(self, c_t_minus1, h_t_minus1, x_t):
        '''
        Forward Pass to the network
        args:
            c_t_minus1: Previous Cell state. Vector Passed from previous time stamp. Act as Local Knowledge. Size equal to  Hidden State
            h_t_minus1: Previous Hidden State. Global Knowledge of Network
            x_t: "N" dimensional Vector fo a word or input. Let us suppose we use GloVe 300, so it will be a 300 dimensional vector
        out:
            current hidden state vector h_t, current_cell_state vector c_t, current output y
        '''
        z = np.row_stack((h_t_minus1,x_t)) # Concatenate the previous hidden state and current input Word

        f = np.dot(self.W_f,z) + self.b_f # Forget
        f = self.sigmoid(f)

        c_intermediate_above = np.multiply(c_t_minus1, f) # Point to point Multiplication of the Forget gate and Previous cell state. First Part of updating Cell state

        c = np.dot(self.W_c,z)+ self.b_c # Cell state
        c = self.tanh(c)

        i = np.dot(self.W_i,z) + self.b_i # intermediate
        i = self.sigmoid(i)

        c_intermediate_below = np.multiply(c, i) # Second portion required cell state

        c_t = np.sum(c_intermediate_above,c_intermediate_below) # Current cell state. It'll be used as c_t_minus_1 for next time stamp

        o = np.dot(self.W_o,z) + self.b_o # Output
        o = self.sigmoid(o)

        h_t = o * c # Current hidden state. It'll be used as h_t_minus_1 for next time stamp

        y = np.dot(self.W_v, h_t) + self.b_v # using this, you can generate a new word, character etc per cell. Many to Many network

        return h_t, c_t, y


    