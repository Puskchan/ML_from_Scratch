import numpy as np

class l1_reg:
    def __init__(self, lamda) -> None:
        self.lamda = lamda

    def __call__(self, weights):
        return self.lamda * np.sum(np.abs(weights))
    
    def derivation(self,weights):
        return self.lamda * np.sign(weights)
    
class l2_reg:
    def __init__(self,lamda) -> None:
        self.lamda = lamda

    def __call__(self, weights):
        return self.lamda * np.sum(np.square(weights))
    
    def derivation(self, weights):
        return self.lamda * 2 * (weights)
    
class ElasticNet:
    def __init__(self, ratio=0.5, lamda=0.1) -> None:
        self.lamda = lamda
        self.ratio = ratio

    def __call__(self, weights):
        l1 = self.ratio * self.lamda * np.sum(np.abs(weights))
        l2 = (1 - self.ratio) * self.lamda * 0.5 * np.sum(np.square(weights))
        return l1+l2
    
    def derivation(self, weights):
        l1 = self.ratio * self.lamda * np.sign(weights)
        l2 = (1 - self.ratio) * self.lamda * (weights)
        return l1+l2