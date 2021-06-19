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


class ScaledDotProductAttention:
    def __init__(self):
        '''
        Last Dimension of Vector or Embedding Dimension of each token/word. for example Glove has 100,200,300 etc dimension size for each "word"
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
        similarity_scores = np.matmul(query, keys.T) # dot product of Query and Keys. Gives similarity of "query" with all the available vectors in that sentence including itself
        scaled_similarity_scores = similarity_scores / np.math.sqrt(query.shape[-1]) # Re-Scale the scores. Formula Given in the paper

        if mask is not None:
            scaled_similarity_scores += (mask * -1e9) # Add the mask. Values present as True/1 will be converted to a very very Huge negative number -1*10^9
        
        normalized_scaled_scores = softmax(scaled_similarity_scores, axis=-1) # normalize the scores
        output = np.dot(normalized_scaled_scores, values) # Multiply the values by the scores and get new weighted values 

        return output, normalized_scaled_scores





