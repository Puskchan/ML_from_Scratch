import numpy as np

class LinearRegression():
    def __init__(self, alpha, iter) -> None:
        self.alpha = alpha # Lerning rate
        self.iter = iter # No of Iterations
        self.weight = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.iter):

            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, y_pred - y)
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db

        return self
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias