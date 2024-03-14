from Regression import Linear_regression
import Regression as Regularization_class


class LassoRegression(Linear_regression.LinearRegression):

    def __init__(self, lamda, learning_rate, iteration):
        self.regularization = Regularization_class.l1_reg(lamda)
        super(LassoRegression, self).__init__(learning_rate, iteration, self.regularization)

    def train(self, X, y):
        return super(LassoRegression, self).fit(X, y)
    
    def predict(self, test_X):
        return super(LassoRegression, self).predict(test_X)