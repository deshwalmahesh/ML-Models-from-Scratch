class Node:
    '''
    A binary tree which has a data element and each node points to right and left elements which are also Node.
    '''
    def __init__(self, data, left = None, right = None):
        '''
        '''
        self.data = data
        self.left = left
        self.right = right

    
    def __str__(self):
        '''
        Print The Node
        '''
        ...

    
    def __repr__(self):
        ...


    def insert(self, data):
        '''
        Insert the data either on the left or right node. If Right or left are present, then it'll keep on traversing all the way down.
        '''
        if data < self.data: # if data is less than left
            if not self.left: # if left is empty, add a new Node on the left side with same data
                self.left = Node(data)
            else: # if left is already present
                self.left.insert(data) # it'll keep on traversing all the way down because each entry is a Node itself
        else:
            if not self.right:
                self.right = Node(data)
            else:
                self.right.insert(data)
    



     