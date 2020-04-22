"""
CSCC11 - Introduction to Machine Learning, Winter 2020, Exam
E. Franco, B. Chan, D. Fleet

===========================================================
 COMPLETE THIS TEXT BOX:

 Student Name:
 Student number:
 UtorID:

 I hereby certify that the work contained here is my own


 ____________________
 (sign with your name)
===========================================================
"""

import numpy as np

from utils import mean_squared_error

class CrossValidation:
    def __init__(self, val_percent=0.3, rng=np.random):
        """ This class represents a PCA subspace with its components and mean computed from data.

        For the following:
        - N: Number of samples.
        - D: Dimension of input features.
        - T: Number of training samples.
        - V: Number of validation samples

        TODO: You will need to implement the methods of this class:
        - _train_validation_split: ndarray, ndarray, float -> ndarray, ndarray, ndarray, ndarray
        - compute_errors: object (PolynomialRegression), ndarray, ndarray, ndarray, ndarray -> float, float

        Implementation description will be provided under each method.

        Args:
        - val_percent (float): The percentage of data held out as the validation set.
                               (1 - val_percent) is the percentage of data for training set.
        - rng (RandomState): The random number generator to permute data.
        """
        assert 1 > val_percent > 0, f"val_percent must be between 0 and 1 exclusively. Got: {val_percent}"

        self.val_percent = val_percent
        self.rng = rng

    def train_validation_split(self, X, y):
        """ This method splits data into 2 random parts, the sizes which depend on val_percent.

        NOTE: For the following:
        - T: Number of training samples.
        - V: Number of validation samples

        Args:
        - X (ndarray (shape: (N, D))): A N-D matrix consisting N D-dimensional vector inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar observed outputs.

        Outputs:
        - train_X (ndarray (shape: (T, D))): A T-D matrix consisting of T-D dimensional training inputs.
        - train_y (ndarray (shape: (T, 1))): A T-column vector consisting of T scalar training outputs.
        - val_X (ndarray (shape: (V, D))): A V-D matrix consisting of V-D dimensional validation inputs.
        - val_y (ndarray (shape: (V, 1))): A V-column vector consisting of V scalar validation outputs.
        """
        N = X.shape[0]
        
        # ====================================================
        # TODO: Implement your solution within the box
        # Create training and validation sets.
        
        # ====================================================

        return train_X, train_y, val_X, val_y

    def compute_errors(self, model, train_X, train_y, val_X, val_y):
        """ This method computes the training and validation errors for a single model.

        NOTE: For the following:
        - T: Number of training samples.
        - V: Number of validation samples

        Args:
        - model (object (PolynomialRegression)): The model to train and evaluate on.
        - train_X (ndarray (shape: (T, D))): A T-D matrix consisting of T-D dimensional training inputs.
        - train_y (ndarray (shape: (T, 1))): A T-column vector consisting of T scalar training outputs.
        - val_X (ndarray (shape: (V, D))): A V-D matrix consisting of V-D dimensional validation inputs.
        - val_y (ndarray (shape: (V, 1))): A V-column vector consisting of V scalar validation outputs.
        
        Output:
        - training_error (float): The training error of the trained model.
        - validation_error (float): The validation error for the trained model.
        """
        # ====================================================
        # TODO: Implement your solution within the box
        # Compute training and validation errors.

        # ====================================================

        return training_error, validation_error
