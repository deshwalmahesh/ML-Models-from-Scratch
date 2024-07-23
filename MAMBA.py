import numpy as np

# First attempt

"""
Code based on copying idea from the working from: https://www.linkedin.com/posts/tom-yeh_deeplearning-neuralnetworks-generatieveai-activity-7142854622003560448-PuSc
Need lots of optimizations and understanding
"""

class Dummy_MAMBA:
    def __init__(self,hidden_state_size = 2):
        self.hidden_state_size = hidden_state_size

        # Need to understand why it is of size [4,4] what does each signify?
        self.input_matrix_weights = np.vstack([np.array([[1,-1,0,0],[0,-1,0,1],[1,0,-1,0],[1,0,0,-1]]),
                                               np.array([[1,0,-1,0],[0,1,0,-1],[1,-1,0,0],[0,0,-1,1]]),
                                               np.array([[-1,0,0,0],[1,0,0,0],[0,0,-1,0],[0,1,0,0]]), 
                                               np.array([[1,-1,0,0],[0,0,-1,1],[1,0,0,0],[0,-1,1,0]])
                                               ]) # learnable. Can initialize with np.random.randint(-1,1,(16,4)) in this specific case
        
        self.h_t =  np.array([0]*self.hidden_state_size) # changes with input and previous state
        self.A_matrix = [np.array([[1, 0], [0, -1]]),
                         np.array([[1, 0], [0, -1]]),
                         np.array([[-1, 0], [0, 1]]),
                         np.array([[-1, 0], [0, 1]])] # what is logic behind this? Why are 1,2 and 3,4 same?

    
    def forward(self,user_input):
        n = 2 # Grouping of 2 is done for the below transformed_input. Need to understand why the transformed is broken into a group of 2? What is logic behind this?

        # Below 4 lines are just optimisations. If we follow the post, it's supposed to be in a loop. 1 Matrix for each input
        transformed_input = self.input_matrix_weights  @ user_input # in the tutorial, it is done one by one so we can do the same though
        lst = transformed_input.copy()
        lst = [lst[i:i + n] for i in range(0, len(lst), n)] 
        transformed_input = [lst[i:i + n] for i in range(0, len(lst), n)]

        y = []
        for index, x in enumerate(user_input):
            A = self.A_matrix[index]
            B = transformed_input[index][0]
            C = transformed_input[index][1]

            h_t_plus1 = (B * np.array([x])) + (A @ self.h_t)

            output = C.dot(h_t_plus1)

            self.h_t = h_t_plus1.copy()

            y.append(output)
        
        return y

mamba = Dummy_MAMBA()
mamba.forward(np.array([3,4,5,6]))
