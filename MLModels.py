import numpy as np
from math import log
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import Counter

SEED = 13
np.random.seed(SEED)


class UnivariantLinearRegression():    
    '''
    Code to implement Linear Regression on 2 variables only i.e Predict 'y' based on 'x' using Loops only. 
    Uses mean, variance, Covariance of X and Y to find b0 and b1 values.
    '''
    def __init__(self):
        self.help_urls = {'intuition':'https://youtu.be/nk2CQITm_eo',
                            'calculate_bo_b1':'https://youtu.be/679fnFowxDk',
                            'practical_implementation':'https://youtu.be/it2Lqu5sS_Y'}


    def calculate_mean(self,values:[list,tuple])->float:
        '''
        Method to calculate the Mean of given Numners. These will be  different 'x' values
        args:
            values: X values in training
        '''
        return sum(values) / float(len(values))


    def calculate_variance(self,values:[list,tuple], mean:[float])->float:
        '''
        Method to Compute the Variance in the given 'X' data points
        args:
            values: X values in the training data
        '''
        return sum([(x-mean)**2 for x in values])


    def calculate_covariance(self,x:[tuple,list],y:[tuple,list],mean_x:float,mean_y:float)->float:
        '''
        Calculate the Covariance of X and Y. How isY affected when X is changed. Is y incresing/decreasing when X is increasing or decreasing
        args: 
            x: X data values
            y: Corresponding Y values
            mean_x: Mean of X
            mean_y: Mean of Y
        '''
        covar = 0.0
        for i in range(len(x)):
            covar += (x[i] - mean_x) * (y[i] - mean_y)
        return covar


    def fit(self,x:[list,tuple],y:[tuple,list])->tuple:
        '''
        Calculate the coefficient bo and b1 based on the given X values
        args:
            x: X values
            y: y values
        '''
        assert len(x) == len(y), "Length of X and Y must be equal"
        x_mean, y_mean = self.calculate_mean(x), self.calculate_mean(y)
        self.b1 = self.calculate_covariance(x, x_mean, y, y_mean) / self.calculate_variance(x, x_mean)
        self.b0 = y_mean - self.b1 * x_mean
    

    def predict(self,x_test:[list,tuple])->list:
        '''
        Return predictions on test data
        '''
        predictions = []
        for x in x_test:
            y_hat = self.b0 + self.b1 * x
            predictions.append(y_hat)
        return predictions


class LinearRegression():
    '''
    Class to apply Linear Regression on Multivariate (multi features) X and predict Y based on MSE Loss
    Uses efficient way of Gradient Descent and Matrix Multiplication method  instead of simple loops 
    ''' 
    def __init__( self,lr:float=0.001,itr:int=500):
        '''
        args:
            lr: Learning rate. It is based on the formula W_curr = W_curr - lr * dW_curr 
            itr: No of iterations to perform
        '''    
        self.lr = lr 
        self.itr = itr 
        self.loss = 0 # it can be mse, rmse or mae
        self.help_links = ['https://youtu.be/4PHI11lX11I',
                            'https://mccormickml.com/2014/03/04/gradient-descent-derivation/',
                            ]


    def predict(self,X:np.ndarray):
        '''
        Predict y_hat for each data point based on y_hat = b0*x + c but for ALL features using Matrix Multiplication
        args:
            X: Numpy array of features. Use this for X_test values for inference
        ''' 
        return X.dot(self.W) + self.b 


    def update_weights(self)->float: 
        '''
        Update the weight and Bias matrix based on the itrative approach of Gradient Descent and return MSE loss for current iteration
        '''  
        Y_pred = self.predict(self.X ) # get y_hat 

        # calculate gradients   
        dW = - (2 * (self.X.T ).dot(self.Y - Y_pred)) / self.m # derivative of Weight matrix. See first link
        db = - 2 * np.sum(self.Y - Y_pred) / self.m  # derivative of bias. See links

        self.W = self.W - self.lr * dW # # update weights based on learning rate
        self.b = self.b - self.lr * db
        return sum((self.Y - Y_pred)**2)/self.m # return MSE loss


    def fit(self,X_train:np.ndarray,y_train:np.ndarray)->None:
        '''
        Fit the data to the model and apply Gradient Descent 
        args:
            X_train: Numpy array of X features
            y_train: Corresponding y values
        '''
        # no_of_training_examples, no_of_features 
        self.X = X_train   
        self.Y = y_train
        self.m, self.n = X_train.shape  # no of data points, No of features 
        self.W = np.zeros(self.n) # weight initialization. Shape is equal to No of features
        self.b = 0 # keep bias to zero at first 
                 
        for _ in range(self.itr): # gradient descent learning
            self.loss = self.update_weights()



class LogisticRegression():
    '''
    Class to apply Logistic Regression on Multivariate (multi features) X and predict Y based on Binary Cross Entropy Loss
    ''' 
    def __init__( self,lr:float=0.001,itr:int=500,thresh:float=0.5,eps:float=1e-15):
        '''
        args:
            lr: Learning rate. It is based on the formula W_curr = W_curr - lr * dW_curr 
            itr: No of iterations to perform
            thresh: Threshold for the condition for the classification. It is supposed to be 0.5 
            eps: Epsilon to be added in the Binary Cross Entropy A.K.A Logg Loss calculation so that the predicted can not be 0
        '''    
        self.lr = lr 
        self.itr = itr
        self.thresh = thresh 
        self.eps = eps
        self.loss = 0 # it has to be Binary Cross Entropy
        self.help_links = {'Cross_Entropy_loss':'https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a',
                            'Odd_LogOdd_Logits':'https://youtu.be/ARfXDSkQf1Y',
                            'Coefficients':'https://youtu.be/vN5cNN2-HWE',
                            'Max_Likelihood':'https://youtu.be/BfKanl1aSG0',
                            'Gradient_Descent':'https://youtu.be/z_xiwjEdAC4'}


    def predict(self,X:np.ndarray)->np.ndarray:
        '''
        Predict y_hat for each data point based on y_hat = b0*x + c but for ALL features using Matrix Multiplication
        args:
            X: Numpy array of features. Use this for X_test values for inference
        ''' 
        A = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) ) # Value of Z will be in between 0 and 1      
        Y_hat = np.where( A > self.thresh, 1, 0 )  # if value of Z is > 0.5, assign it to class 1 else assign it to class 0       
        return A,Y_hat  


    def update_weights(self): 
        '''
        Update the weight and Bias matrix based on the itrative approach of Gradient Descent
        '''  
        A,Y_hat = self.predict(self.X) 
          
        # calculate gradients         
        tmp = (A - self.Y.T )         
        tmp = np.reshape( tmp, self.m )         
        dW = np.dot( self.X.T, tmp ) / self.m          
        db = np.sum( tmp ) / self.m  
          
        # update weights     
        self.W = self.W - self.lr * dW     
        self.b = self.b - self.lr * db
        return A,Y_hat # return loss 


    def fit(self,X_train:np.ndarray,y_train:np.ndarray)->None:
        '''
        Fit the data to the model and apply Gradient Descent 
        args:
            X_train: Numpy array of X features
            y_train: Corresponding y values
        '''
        # no_of_training_examples, no_of_features 
        self.X = X_train   
        self.Y = y_train
        self.m, self.n = X_train.shape  # no of data points, No of features 
        self.W = np.zeros(self.n) # weight initialization. Shape is equal to No of features
        self.b = 0 # keep bias to zero at first 
                 
        for _ in range(self.itr): # gradient descent learning
            dist,_ = self.update_weights()
            self.loss = -sum([y_train[i] * log(dist[i] + self.eps) for i in range(len(y_train))])


class LDA():
    '''
    Class to Perform Linear Discriminant Analysis or LDA. Returns FIRST 2 Components only.
    Aims to increase the Data Points distance between different classes and decrease the distance between same class.
    ''' 
    def __init__(self,df:pd.DataFrame,label_col:str):
        '''
        args:
            df: Dataframe which has Data Points (Features) as well as the Class Column
            label_col: Name of the column which contains the Labels / Classes
        '''
        assert (isinstance(df,pd.DataFrame)), "Input should be a DataFrame which has Features + class/labels columns"
        self.df = df
        self.class_name = label_col
        self.n = df.shape[1]-1


    def find_class_vise_mean(self)->pd.DataFrame:
        '''
        Find and Return the Mean of EACH class.
        '''
        class_vise_mean = self.df.groupby(self.class_name).mean().T
        return class_vise_mean
    
    
    def find_within_class_scatter(self):
        '''
        Fnnd the Scatter Matrix within  EACH clasS
        '''
        class_vise_mean = self.find_class_vise_mean()
        within_class_scatter_matrix = np.zeros((self.n,self.n))
        for class_, rows in self.df.groupby('class'):
            rows = rows.drop([self.class_name], axis=1)
            dot_product = np.zeros((self.n,self.n))
            class_mean = class_vise_mean[class_].values.reshape(self.n,1)

            for _, row in rows.iterrows():
                n_th_row = row.values.reshape(self.n,1) 
                # get all the elements in the columns row-vise that belong to current class in a form of 2-D array
                dot_product += (n_th_row - class_mean).dot((n_th_row - class_mean).T)

                # for each column element 'X', subtract it's 'own' class mean for all the x_i
                # i.e for a row element say 'flower' at index 13, it has 4 ATTRIBUTES corresponding to sepal and 
                # petal's width and heights and suppose it belongs to class 2. for each ATTRIBUTE in this flower,
                # subtract the attribute from the class' mean it belong to i.e class-2 mean because each has
                # individual mean for each ATTRIBUTE. Get a Transpose to get a DOT*

            within_class_scatter_matrix += dot_product
        return within_class_scatter_matrix
    
    
    def find_bw_class_scatter(self):
        '''
        Find the Scatter Matrix AMONG classes
        '''
        class_vise_mean = self.find_class_vise_mean()
        feature_means = self.df.drop(self.class_name,axis=1).mean() # means of individual features/columns
        between_class_scatter_matrix = np.zeros((self.n,self.n))

        for class_ in class_vise_mean:    
            total_elements_in_class = len(self.df.loc[self.df[self.class_name] == class_].index)

            class_m = class_vise_mean[class_].values.reshape(self.n,1)
            feat_m = feature_means.values.reshape(self.n,1)
            # mean that belongs to current class(0,1,2), mean of individual features(sep_h,pet_l,sep_l....)

            between_class_scatter_matrix += total_elements_in_class * (class_m - feat_m).dot((class_m - feat_m).T)
        return between_class_scatter_matrix
    
    
    def get_eign_value_vector(self)->tuple:
        '''
        Get ALL Eign Values and Vectors corresponding to the data
        '''
        within_class_scatter_matrix = self.find_within_class_scatter()
        between_class_scatter_matrix = self.find_bw_class_scatter()
        eign_values, eign_vectors = np.linalg.eig(np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))
        return (eign_values,eign_vectors)
    
    
    def find_eign(self,display_only:bool=False):
        '''
        Get the explained Variance described by the Eign Vectors and Values
        args:
            display_only: Whether to just show the explained variance or return the Eign Vector - Value Pairs
        '''
        eign_values,eign_vectors = self.get_eign_value_vector()
        eign_value_vector_pair = [(np.abs(eign_values[i]), 
                               eign_vectors[:,i]) for i in range(len(eign_values))]
        eign_value_vector_pair = sorted(eign_value_vector_pair, key=lambda x: x[0], reverse=True)
        eign_value_sums = sum(eign_values)
        
        if display_only:
            print('Explained Variance by each EignVector in terms of total info')
            for i, pair in enumerate(eign_value_vector_pair):
                print('{}: {:.2f}%'.format(i+1, (pair[0]/eign_value_sums).real*100))
            return None
        else:
            return eign_value_vector_pair
      
        
    def fit_transform(self):
        '''
        Fit the Data and return the LDA values
        '''
        X = self.df.drop(self.class_name,axis=1).values
        eign_value_vector_pair = self.find_eign()
        W_matrix = np.hstack((eign_value_vector_pair[0][1].reshape(self.n,1), 
                      eign_value_vector_pair[1][1].reshape(self.n,1))).real # run this in a loop for n_components
        lda = np.array(X.dot(W_matrix))
        return lda
    
    
    def plot(self):
        '''
        Plot 2-D Scatter plot of LDA values colored by Class Names / Labels
        '''
        lda = self.fit_transform()
        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.scatter(lda[:,0],lda[:,1],c=self.df[self.class_name],
                    cmap='rainbow',alpha=0.7,edgecolors='b')


class PCA():
    '''
    Apply PCA or Principal Component Analysis for unsupervised dimensionality reduction. Uses Eign values, vectors to find the directions which
    preserves the maximum variance in the data.
    Provide the Scaled Data to PCA to get the most out of it. USe StandardScaler() class from sklearn.preproccessing.
    '''
    def __init__(self,n_components:int):
        '''
        n_components: Number of components to find. 0 < n_components <= no of features
        '''
        self.n_components = n_components


    def fit(self,X:np.ndarray,transform:bool=True):
        '''
        Find the Covariance Matrix -> Find Eign vectors and Values -> 
        args:
            X: Numpy array of features to be fitted on the data
            transform: Whether to fit only or transform too
        '''
        assert 0 < self.n_components <= X.shape[1], "Provide a value such that 0 < n_components <= X.shape[1]"
        cov_matrix = np.cov(X.T) # Transpose the features matrix for finding the 
        self.values, self.vectors = np.linalg.eig(cov_matrix)  # np.linalg.eigh (see 'h' in end) would give different eign values and vectors

        self.explained_variances = [] # Explained Variance
        for i in range(len(self.values)):
            self.explained_variances.append(self.values[i] / np.sum(self.values))
        
        if transform:
            return self.transform(X)


    def transform(self, X:np.ndarray)->np.ndarray:
        '''
        Transform the Data based on the values provided
        args:
            X: Transform the X features based on the found Eign Values and vectors
        '''
        projected_components = []
        for i in range(self.n_components):
            projected_components.append(X.dot(self.vectors.T[i]))
        return np.array(projected_components).T


class KNN:
    def __init__(self,k:int=5):
        '''
        args:
            k: No of data points to consider as neighbours
        '''
        self.k = k


    def minkowski_distance(self,a:np.ndarray,b:np.ndarray,p:int=2)->float:
        '''
        Get the distance between two points based on Minkowski distance. It is just a generalised form of many distance given value of 'p'.
        For p = 2, it'll give Euclidean distance and for p = 1, it'll give Manhattan distance.

        args:
            a: array of features
            b: array of features
            p: distance specifier
        '''
        return np.power(np.sum(np.power(np.abs(a-b),p)),1/p)
    

    def fit(self,X_train:np.ndarray,y_train:np.ndarray):
        '''
        Fit the Given data to the model
        X_train: X features
        y_train: Corresponding Y labels
        '''
        self.X_train = X_train
        self.y_train = y_train
    

    def classify(self,x_test:np.ndarray)->int:
        '''
        Return the distance of a SINGLE data point to whole of the training data and produce Y label
        x_test: 1-D Numpy array of Single data point at a time
        '''
        distances = [self.minkowski_distance(x_test,x_each) for x_each in self.X_train] # Get distances for each point in train data
        top_k_indices = np.argsort(distances[:self.k]) # Get indices of top K data points which are the closest
        top_k_labels = [self.y_train[i] for i in top_k_indices] # Get the Y labels of each TOP-K point who are the closest
        return Counter(top_k_labels).most_common(1)[0][0] # First element in labels is the most frequently occured element


    def predict(self,X_test:np.ndarray)->np.ndarray:
        '''
        Predict on an incoming dataset
        X_test: Test Data freatures
        '''
        return np.array([self.classify(x) for x in X_test])

