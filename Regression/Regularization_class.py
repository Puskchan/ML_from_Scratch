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