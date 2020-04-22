"""
CSCC11 - Introduction to Machine Learning, Winter 2020, Exam
B. Chan, E. Franco, D. Fleet

===========================================================
 COMPLETE THIS TEXT BOX:

 Student Name: Abhishek Chatterjee
 Student number: 1004820615
 UtorID: chatt114

 I hereby certify that the work contained here is my own


 _Abhishek Chatterjee_
 (sign with your name)
===========================================================
"""

import numpy as np

class PolynomialRegression:
    def __init__(self, K):
        """ This class represents a polynomial regression model which is similar 
        to Assignment 1, but we want this to work with multi-dimensional inputs.
           
        self.parameters contains the model weights.
        NOTE: We have no requirement regarding to the order of parameters.
              e.g. the bias term does not need to be the first element of self.parameters.

        NOTE: Fit the model before calling predict.

        TODO: You will need to implement the methods of this class:
        - transform_matrix: ndarray -> ndarray
        - predict: ndarray -> ndarray
        - fit: ndarray -> None

        Implementation description will be provided under each method.
        
        For the following:
        - N: Number of samples.
        - D: Dimension of input feature vectors.
        - K: Degree of polynomial model.
        - W: Number of parameters (including bias).

        Args:
        - K (int): The degree of the desired polynomial model. Note: K  is either 1 or 2.
        """
        assert 1 <= K <= 2, f"polynomial order K must be 1 or 2. Got: {K}"
        self.K = K

        self.parameters = None

    def transform_matrix(self, X):
        """  This method transforms matrix of N inputs (feature vectors) into 
        the polynomial basis-function matrix (the B matrix in course notes).

        NOTE: We will not be checking the shape of B for you.
              Please make sure the shape is correct (i.e. how many terms should we have?).

        Args:
        - X (ndarray (shape: (N, D))): NxD matrix consisting of N D-dimensional feature vectors

        Output:
        - B (ndarray (shape: (N, W))): NxW matrix consisting of N W-dimensional transformed feature
                                       vectors representing the polynomial basis function matrix.
        """
        (N, D) = X.shape

        # ====================================================
        # TODO: Implement your solution within the box

        # B matrix is Vandermonde matrix with feature vectors 

        #B = np.vander(X.flatten(), self.K+1, True)

        # ====================================================
           
        return B

    def predict(self, X):
        """  This method takes as input a matrix of input feature vectos and outputs predicted values 
        using the polynomial regression model.        

        Args:
        - X (ndarray (shape: (N, D))): NxD matrix consisting of N D-dimensional feature vectors.

        Output:
        - pred (ndarray (shape: (N, 1))): A N-column vector consisting N scalar output data.
        """
        assert len(X.shape) == 2, f"X must be a 2D array. Got: {X.shape}"

        # ====================================================
        # TODO: Implement your solution within the box

        #B = transform_matrix(self, X)   # check if this behaves as expected
        #pred = np.matmul(B, self.parameters)

        # ====================================================

        assert pred.shape == (X.shape[0], 1), f"dimensionality of predictions is not correct. Expected: {(X.shape[0], 1)}. Got: {pred.shape}"
        return pred

    def fit(self, train_X, train_y):
        """ This method fits the model parameters, given the training inputs and outputs.

        NOTE: You will need to replace self.parameters with the computed optimal parameters. 
		Remember that the shape of the self.parameters is (W, 1).

        NOTE: We will not be checking the shape of self.parameters for you.
              Please make sure the shape is correct (i.e. how many parameters should we have?).

        Args:
        - train_X (ndarray (shape: (N, D))): A N-D matrix consisting of N-D dimensional training inputs.
        - train_y (ndarray (shape: (N, 1))): A N-column vector consisting of N scalar training outputs.
        """
        assert len(train_X.shape) == len(train_y.shape) == 2 and train_X.shape[0] == train_y.shape[0], f"input and/or output has incorrect shape (train_X: {train_X.shape}, train_y: {train_y.shape})."

        # ====================================================
        # TODO: Implement your solution within the box

        #B = transform_matrix(self, train_X)     # check if this behaves as expected
        #p1 = np.linalg.inv(np.matmul(B.T, B))
        #p2 = np.matmul(B.T, train_y)

        #self.parameters = np.matmul(p1, p2)

        # ====================================================
