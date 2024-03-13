import numpy as np
from itertools import combinations_with_replacement
from Regression import Linear_regression

def PolynomialFeature(X, degree):
    n_samples, n_features = X.shape
    combination = [combinations_with_replacement(range(n_features), i) for i in range(0, degree+1)]
    combination_index = [index for obj in combination for index in obj]

    new_features = len(combination_index)
    X_new = np.empty((n_samples, new_features))

    for i, com_idx in enumerate(combination_index):
        X_new[:, i] = np.prod(X[:, com_idx], axis=1)

    return X_new

class PolynomialRegression(Linear_regression.LinearRegression):
    def __init__(self, alpha, iter, degree, regularization=None) -> None:
        self.degree = degree
        self.regularization = lambda x:0
        self.regularization.derivation = lambda x:0
        super().__init__(alpha, iter, self.regularization)
    
    def train(self, X, y):
        X_poly = PolynomialFeature(X, degree=self.degree)
        return super().train(X_poly, y)
    
    def predict(self, test_X):
        test_X_poly = PolynomialFeature(test_X, degree=self.degree)
        return super().predict(test_X)
