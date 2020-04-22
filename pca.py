"""
CSCC11 - Introduction to Machine Learning, Winter 2020, Exam
B. Chan, S. Wei, E. Franco, D. Fleet

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

import matplotlib.pyplot as plt
import numpy as np

class PCA:
    def __init__(self, X):
        """ This class represents PCA with components and mean computed from data.

        TODO: You will need to implement the following methods of this class:
        - plot_variance_per_subspace: ndarray -> ndarray
        - plot_fraction_variance: ndarray -> ndarray

        Implementation description will be provided under each method.

        For the following:
        - N: Number of samples.
        - D: Dimension of input feature vectors.
        - K: Dimension of low-dimensional representation of input features.
             NOTE: K >= 1

        Args:
        - X (ndarray (shape: (N, D))): NxD matrix consisting of N D-dimensional input vectors on its rows.
        """

        # Mean of each column, shape: (D, )
        self.mean = np.mean(X, axis=0)
        self.V, self.w = self._compute_components(X)
        self.D = self.mean.shape[0]

    def _compute_components(self, X):
        """ This method computes the PCA directions (one per column) given data.

        NOTE: Use np.linalg.eigh to compute the eigenvectors

        Args:
        - X (ndarray (shape: (N, D))): NxD matrix consisting N D-dimensional input data.

        Output:
        - V (ndarray (shape: (D, D))): DxD matrix of PCA eigen directions (one per column) sorted according to the eigenvalues in descending order.
        - w (ndarray (shape: (D, ))): D-column vector of the eigenvalues of the PCA directions sorted in descending order.
        """
        assert len(X.shape) == 2, f"X must be a NxD matrix. Got: {X.shape}"
        (N, D) = X.shape

        # find eigenvalues and eigenvectors of the covariance matrix
       	data_shifted = X - self.mean
        data_cov = np.cov(data_shifted, rowvar=False)
        w, V = np.linalg.eigh(data_cov)
        w = np.flip(w)
        V = np.flip(V, axis=1)

        assert V.shape == (D, D), f"V shape mismatch. Expected: {(D, D)}. Got: {V.shape}"
        assert w.shape == (D,), f"w shape mismatch. Expected: {(D,)}. Got: {w.shape}"
        return V, w

    def reduce_dimensionality(self, X, K):
        """ This method maps data X onto a K-dimensional subspace using precomputed mean and PCA components.

        Args:
        - X (ndarray (shape: (N, D))): NxD matrix consisting N D-dimensional input data.
        - K (int): Number of dimensions for the low-dimensional subspace model.

        Output:
        - low_dim_X (ndarray (shape: (N, K))): NxD matrix with N K-dimensional vectors that represent input data.
        """
        assert len(X.shape) == 2, f"X must be a NxD matrix. Got: {X.shape}"
        (N, D) = X.shape
        assert D == self.D, f"dimensionality of representation must be {self.D}. Got: {D}"
        assert self.D >= K > 0, f"dimensionality of lower dimensionality representation must be between 1 and {self.D}. Got: {K}"

        low_dim_X = (X - self.mean) @ self.V[:, :K]

        assert low_dim_X.shape == (N, K), f"low_dim_X shape mismatch. Expected: {(N, K)}. Got: {low_dim_X.shape}"
        return low_dim_X

    def reconstruct(self, low_dim_X):
        """ This method reconstructs X from the low-dimensional PCA subspace representation using precomputed mean and principal components.

        NOTE: The dimension K is implicitly specified by low_dim_X.

        Args:
        - low_dim_X (ndarray (shape: (N, K))): NxD matrix consisting N K-dimensional vectors representing N data points in the subspace.

        Output:
        - X (ndarray (shape: (N, D))): NxD matrix consisting N D-dimensional reconstructed data points.
        """
        assert len(low_dim_X.shape) == 2, f"low_dim_X must be a NxK matrix. Got: {low_dim_X.shape}"
        (N, K) = low_dim_X.shape
        assert K > 0, f"dimensionality of representation must be at least 1. Got: {K}"

       	X = low_dim_X @ self.V[:, :K].T + self.mean

        assert X.shape == (N, self.D), f"X shape mismatch. Expected: {(N, self.D)}. Got: {X.shape}"
        return X

    def plot_variance_per_subspace(self):
        """ This function plots the variance captured by each subspace dimension from 1 to D.

        Output:
        - variances (ndarray (shape: (D,))): D-column vector corresponding to the variances captured by each subspace dimension.
        """
        
        # ====================================================
        # TODO: Implement your solution within the box

        # ====================================================
        assert variances.shape == (self.D,), f"variances shape mismatch. Expected: {(self.D,)}. Got: {variances.shape}"

       	plt.plot(np.arange(1, self.D + 1), variances, marker="o")
        plt.title("Variance per Subspace Dimension")

        plt.show()
        plt.clf()

        return variances

    def plot_fraction_variance(self):
        """ This function plots the fraction of the total variance in the data as a function of the dimension of the subsapce model, from 1 to D.

        NOTE: Include the case when K=0.

        Output:
        - fractions (ndarray (shape: (D,))): D-column vector corresponding to the fractions of the total variance.
        """

        # ====================================================
        # TODO: Implement your solution within the box

        # ====================================================
        assert fractions.shape == (self.D + 1,), f"fractions shape mismatch. Expected: {(self.D + 1,)}. Got: {fractions.shape}"

       	plt.plot(np.arange(0, self.D + 1), fractions, marker="o")
        plt.title("Fractions of Total Variance")

        plt.show()
        plt.clf()

        return fractions
