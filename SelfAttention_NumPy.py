import numpy as np

def softmax(X:np.ndarray, axis:int=-1)->np.ndarray:
    '''
    return the normalized values from the input values
    args:
        X: Input data of shape (batch_size, No of features)
        axis: Which axis to work with. Default is last axis
    '''
    # axis=1, keepdims=True parameter is used as we work with batches not single input so the correct value is operated from itself 
    max_value_per_instance = np.max(X, axis=axis, keepdims=True) # Get max for each input and Center the Values so that you don't blow the exponential as Max will be 0 and others will be -ive
    centered_exp = np.exp(X - max_value_per_instance) # Get the exponential values for each data point
    return centered_exp / np.sum(centered_exp, axis=axis, keepdims=True) # apply the formula as e^x / sum(e^x) for each input in the batch


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
            self.output = np.dot(X, self.W) + self.b # y = W.X + b  (X dot (W_Transpose)) 


class ScaledDotProductAttention:
    '''
    Self-Attention: A method to "re-weight" the given Embeddings of words based on the relation with all the other words in the sentence. Usually a "quey" a is picked from sentence, dot product
    of query is calculate with all the other words in that sentence which gives a score. Now that score is multiplied to values so that there can be a "temporary" embedding for each based
    based on what is the "current query" 
    '''
    def forward(self, query, keys, values, mask=[np.ndarray,None]):
        '''
        args:
            query: Vector (vectorised token) whose similarity has to be found out using all the other vectors in the sentence. Query is usually chosen from the key
            keys: Whole Vectorised Sentence whose similarity has to be found out with key. Usually query is part of the keys (one of the vectorised token of keys)
            values: Self-Attention score has to be multiplied by this vector. Usually Values are same as Keys in Self Attention
            mask: Mask of boolean array. Used in translation task when we don't want network to see the words present "in future"
        out:
            return re-weighted values, Normalised scaled scores
        '''
        assert query.shape[-1] == keys.shape[-1], "query and Keys must have length of  EMBEDDING Dimension"
        assert keys.shape[-2] == values.shape[-2], "Both Keys and Values must have the same 'max_len'. Embedding dimension can be different"
        
        similarity_scores = np.matmul(query, keys.T) # dot product of Query and Keys. Gives similarity of "query" with all the available vectors in that sentence including itself
        # np.transpose(keys, [0,2,1]) # test with batch
        scaled_similarity_scores = similarity_scores / np.math.sqrt(query.shape[-1]) # Re-Scale the scores. Formula Given in the paper

        if mask is not None:
            scaled_similarity_scores += (mask * -1e9) # Add the mask. Values present as True/1 will be converted to a very very Huge negative number -1*10^9
        
        normalized_scaled_scores = softmax(scaled_similarity_scores, axis=-1) # normalize the scores
        output = np.matmul(normalized_scaled_scores, values) # Multiply the values by the scores and get new weighted values 

        return output, normalized_scaled_scores


class MultiHeadAttention:
    '''
    Implement Transformer architecture. It is just the advanced version of Self Attention or Scaled Dot product attention where some weights are associated with key, queries, values.
    Also, there is concept of heads. Each head will calculate attention on different words. "She gave her dog some food whose name is Jack": each head will try to mimick and tell that
    there are multiple relation and point of interests. For example She:Jack, She:dog, she:gave, she:food, dog:food, dog:jack, jack:food etc are related so ateention will be calculated from different heads.
    '''
    def __init__(self,heads:int, hidden_size:int, max_len:int):
        '''
        args:
            heads: No of heads used. Each head will calculate the Self-Attention query differently. Sentence is divided in "No of heads"
            hidden_size: Size of the Hidden layers used in Heads. Each of Key, Query, Values have 3 different weights matrices but size is same
            max_len: Maximum length of sentences. Max No of Words present in a single sentence
        '''
        assert hidden_size % heads == 0, "No of heads should be perfectly divisible by dimension size"
        self.depth = hidden_size // heads
        self.heads = heads
        self.hidden_size = hidden_size
        self.max_len = max_len

        self.DQ = Dense(self.hidden_size, self.max_len)
        self.DK = Dense(self.hidden_size, self.max_len)
        self.DV = Dense(self.hidden_size, self.max_len)

        final_layer = Dense(self.hidden_size, self.max_len)

        
    def forward(self,):
        pass
        

        




    






