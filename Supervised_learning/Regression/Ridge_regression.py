from Regression import Linear_regression
import Regression as Regularization_class


class RidgeRegression(Linear_regression.LinearRegression):

    def __init__(self, lamda, learning_rate, iteration):
        self.regularization = Regularization_class.l2_reg(lamda)
        super(RidgeRegression, self).__init__(learning_rate, iteration, self.regularization)

    def train(self, X, y):
        return super(RidgeRegression, self).fit(X, y)
    
    def predict(self, test_X):
        return super(RidgeRegression, self).predict(test_X)