import numpy as np
import pandas as pd
from collections import Counter



def distance_measure(x1,x2,choice="Euclidean",p=1):
    '''

    Choices : Euclidean, Manhattan, Minkowski
    
    '''
    d = 0
    if choice == 'Euclidean':
        for i in range(len(x1)):
           d += np.square(x1[i] - x2[i])
        return np.square(d)
    if choice == 'Manhattan':
        for i in range(len(x1)):
            d += np.abs(x1[i] - x2[i])
        return d
    if choice == 'Minkowski':
        for i in range(len(x1)):
            d += np.power(np.abs(x1[i]-x2[i]), p)
        return np.power(d,1/p)
    
class KNN:
    
    def __init__(self, n_neighbours, choice='Euclidean'):
        self.n_neighbours = n_neighbours
        self.choice = choice

    def most_frequent_class(self, neighbours_y):
        counts = np.bincount(neighbours_y)
        return counts.argmax()
    
    def train(self, X, y):

        self.m, self.n = X.shape
        self.trainX = X
        self.trainy = y

    def predict(self, testX):

        y_pred = np.empty((testX.shape[0],1))

        for index, test_x in enumerate(testX):
            distance = [distance_measure(test_x, train_x ,self.choice) for train_x in self.trainX]

            n_neighbours_index = np.argsort(distance)[: self.n_neighbours]

            n_neighbours_y = np.array([self.trainy[ind][0] for ind in n_neighbours_index ])

            y_pred[index] = self.most_frequent_class(n_neighbours_y)

        return y_pred