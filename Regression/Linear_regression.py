import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, alpha, iter, regularization=None) -> None:
        self.alpha = alpha # Lerning rate
        self.iter = iter # No of Iterations
        self.weight = None
        self.bias = None
        self.y_pred = None
        self.regularization = regularization
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.random.rand(n_features)
        self.bias = 0
        self.y_pred = 0

        for _ in range(self.iter):

            self.y_pred = np.dot(X, self.weight) + self.bias

            dw = ((1/n_samples) * np.dot(X.T, self.y_pred - y))
            if self.regularization:
                dw += self.regularization.derivation(self.weight)
            db = (1/n_samples) * np.sum(self.y_pred-y)

            self.weight -= self.alpha * dw
            self.bias -= self.alpha * db

        return self
    
    def predict(self, test_X):
        return np.dot(test_X, self.weight) + self.bias
    
    def show_line(self,X ,y ,y_pred):
        '''
        X - From the dataset
        y - From the dataset
        y_pred - Declared in the init fn
        '''
        plt.scatter(X, y, color = 'blue' ) 
      
        plt.plot(X, y_pred, color = 'orange' ) 
      
        plt.title( 'Line adjusted to data' ) 
      
        plt.xlabel( 'X' ) 
      
        plt.ylabel( 'Y' ) 
      
        plt.show() 

        return self

if __name__ == '__main__':
    pass