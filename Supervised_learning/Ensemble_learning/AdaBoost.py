import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost():
    def __init__(self, iterations):
        self.weak_classifiers = {}
        self.iterations = iterations

    def upper_lower_range(self,w):
        upper, lower = np.zeros(w.shape[0]), np.zeros(w.shape[0])
        upper = np.cumsum(w)
        lower = w - upper
        return upper,lower

    def create_new_data(self, X_train, w):
        indices = []
        u, l = self.upper_lower_range(w)
        i = 0

        while len(indices)<len(w):
            a = np.random.random()
            if u[i]>a and a>l[i]:
                indices.append(i)
                i += 1
        
        return indices



    def fit(self, X_train, y_train):
        X_copy = X_train
        Y_copy = y_train
        N = len(y_train)
        w = np.full(N, 1/N)

        for _ in range(self.iterations):
            wc = DecisionTreeClassifier(max_depth=1)
            wc.fit(X_copy, Y_copy)

            prediction = wc.predict(X_copy)
            

            errors = (prediction != Y_copy).astype(int)
            weighted_error = np.dot(w, errors) / np.sum(w)
            

            alpha = 0.5 * np.log((1 - weighted_error) / (weighted_error+0.000000000001))
            

            for i in range(len(Y_copy)):
                if prediction[i] == Y_copy[i]:
                    w[i] *= np.exp(-alpha)
                else:
                    w[i] *= np.exp(alpha)
            
            w /= np.sum(w)
            

            idx = self.create_new_data(X_copy, w)
            X_copy = 0
            Y_copy = 0      
            X_copy = X_train[idx,:]
            Y_copy = y_train[idx]
            self.weak_classifiers[wc] = alpha 

    def predict(self, X_test):
        final_pred = np.zeros(X_test.shape[0])
        for wc, alpha in self.weak_classifiers.items():
            final_pred += alpha * wc.predict(X_test)
        return np.sign(final_pred)