from Regression import Linear_regression
from Regression import Regularization_class

class ElasticNet_reg(Linear_regression.LinearRegression):
    def __init__(self, ratio, lamda, lr, iter) -> None:
        self.regularization = Regularization_class.ElasticNet(ratio, lamda)
        super().__init__(lr, iter, self.regularization)

    def train(self, X, y):
        return super().fit(X,y)
    
    def predict(self, test_X):
        return super().predict(test_X)