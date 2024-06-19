import numpy as np
import math

class LogisticRegression():
    def __init__(self, alpha, iter, regularization=None) -> None:
        self.alpha = alpha # Lerning rate
        self.iter = iter # No of Iterations
        self.weight = None
        self.bias = None
        self.regularization = regularization

    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.iter):

            y_pred = 1/(1+math.e**(-(np.dot(X, self.weight) + self.bias)))

            cost = -1/n_samples * (np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred)))

            dw = ((1/n_samples) * np.dot(X.T, y_pred - y))
            if self.regularization:
                dw += self.regularization.derivation(self.weight)
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weight -= self.alpha * dw
            self.bias -= self.alpha * db

            if _ % 100 == 0:
                print(f'Iteration: {_} --- Cost: {cost}')

        return self
    
    def predict(self, test_X):
        y_new = 1/(1+math.e**(-(np.dot(test_X, self.weight) + self.bias)))
        return np.where(y_new > 0.5,1,0)