"""
CSCC11 - Introduction to Machine Learning, Winter 2020, Exam
E. Franco, B. Chan, D. Fleet

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
import matplotlib.pyplot as plt

from cross_validation import CrossValidation
from pca import PCA
from polynomial_regression import PolynomialRegression
from utils import load_pickle_dataset

def model_selection(model, X, y, seed=0, title='PCS vs Cross Validation Scores', visualize_variance=False):
    """ This function helps select a regression model for the input data.  For a given polynomial degree model,
        we need to select the corresponding dimension (K) of the linear subspace (found with PCA) that provides 
 	    the low-dimensional regressor input.

	It generates the following plots:
	1. Variance captured by each subspace in PCA.
	2. Fractions of total variance captured in the PCA subspace as a function of subspsace dimension from 1 to D.
	3. The training/validation error curves, where the x-axis corresponds to the number of principal components used,
	   and the y-axis corresponds to the error.

    	It also displays the best cross validation score and the corresponding number of principal components for that model.

	TODO: In this function, you will need to implement the following:
	- Receive the training and validation errors over all number of principal components using the CrossValidation object for the linear and quadratic models. 
	- Compute the best cross validation score and the corresponding number of principal components used.

	NOTE: You will need to finish polynomial_regression.py, pca.py and cross_validation.py before completing this function.

    Args:
    - X (ndarray (shape: (N, D))): A N-D matrix consisting of N-D dimensional inputs.
    - y (ndarray (shape: (N, 1))): A N-column vector consisting of N scalar outputs.
    - model (object (Polynomial Regression)): The model to train and evaluate.
    - cv (object (Cross Validation)): The cross validation object that splits data into training and validation sets.
    """
    D = X.shape[1]
    training_errors = []
    validation_errors = []
    
    cv = CrossValidation(val_percent, np.random.RandomState(seed))
    train_X, train_y, val_X, val_y = cv.train_validation_split(X, y)

    # ====================================================
    # TODO: Implement your solution within the box
    # Apply PCA to training and validation sets
    # NOTE: Assign the PCA object to a variable called pca (as used in the if statement below).
    # NOTE: You should only apply PCA once here and not in the loop below.

    pca = PCA(train_X)
    pca._compute_components(train_X)                               # updates components

    # ====================================================

    if visualize_variance:
        pca.plot_fraction_variance()
        pca.plot_variance_per_subspace()
    
    for reduce_dim in range(1, D + 1):
        # ====================================================
        # TODO: Implement your solution within the box
        # Receive training and validation errors

        training_error, validation_error = cv.compute_errors(model, pca.reduce_dimensionality(train_X, reduce_dim), train_y, pca.reduce_dimensionality(val_X, reduce_dim), val_y)

        # ====================================================

        training_errors.append(training_error)
        validation_errors.append(validation_error)

    # ====================================================
    # TODO: Implement your solution within the box
    # Assign cv_scores and compute index of the best cross validation score.

    cv_scores = []

    for i in range(D):
        cv_scores.append((training_errors[i] + validation_errors[i]) / 2)   # Use mean as CV score formula

    best_cv_idx = cv_scores.index(min(cv_scores))
    # ====================================================

    print(f"Model: {title}")
    print('Best Cross Validation Score: {}'.format(cv_scores[best_cv_idx]))
    print('Number of Principal Components with Best Cross Validation Score: {}'.format(best_cv_idx + 1))

    # Plot error curves
    range_x = range(1, D + 1)
    plt.plot(range_x, training_errors, label="Training Error", marker="o")
    plt.plot(range_x, validation_errors, label="Validation Error", marker="o")
    plt.title(title)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cross Validation MSE Scores')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dataset = "./data/oil_500.pkl"
    
    rates = load_pickle_dataset(dataset)

    X = rates['X']
    y = rates['y']

    seed = 0
    val_percent = 0.3

    # Linear regression
    model = PolynomialRegression(1)
    model_selection(model, X, y, seed, 'Linear Regression', True)

    # Quadratic regression
    model = PolynomialRegression(2)
    model_selection(model, X, y, seed, 'Quadratic Regression', False)
