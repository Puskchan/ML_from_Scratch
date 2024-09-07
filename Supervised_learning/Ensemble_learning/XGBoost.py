import math
import numpy as np 
import pandas as pd
from collections import defaultdict

class XGBoostModel():
    '''XGBoost from Scratch

    params : 'subsample': 1.0,
             'learning_rate': 0.3,
             'base_score': 0.5,
             'max_depth': 5,
             => Values set by default
    '''
    
    def __init__(self, params, random_seed=None):
        # Setting default values for parameters
        default_params = {
            'subsample': 1.0,
            'learning_rate': 0.3,
            'base_score': 0.5,
            'max_depth': 5,
        }

        # Use the given params or default if not present
        self.params = {**default_params, **params}

        # Assigning individual parameters
        self.subsample = self.params['subsample']
        self.learning_rate = self.params['learning_rate']
        self.base_prediction = self.params['base_score']
        self.max_depth = self.params['max_depth']

        # Random number generator
        self.rng = np.random.default_rng(seed=random_seed)


    def fit(self, X, y, objective, num_boost_round):
        #Initialize predictions with base score
        current_predictions = np.zeros_like(y) + self.base_prediction
        self.boosters = []

        for _ in range(num_boost_round):
            # Calculate gradients and hessians
            gradients = objective.gradient(y, current_predictions)
            hessians = objective.hessian(y, current_predictions)

            # Subsample data if subsample < 1.0
            sample_idxs = self._get_subsample_idxs(y)

            # Train new tree booster
            booster = TreeBooster(X, gradients, hessians,
                                  self.params, self.max_depth, sample_idxs)
            
            # Update Predictions with new tree
            current_predictions += self.learning_rate * booster.predict(X)
            
            # Store the trained booster
            self.boosters.append(booster)

    
    def predict(self, X):
        """Generate predictions for input X."""

        return (self.base_prediction + self.learning_rate
                * np.sum([booster.predict(X) for booster in self.boosters], axis=0))
    
    
    def _get_subsample_idxs(self, y):
        """Helper method for subsampling indices."""
        
        if self.subsample == 1.0:
            return None
        return self.rng.choice(len(y),
                                     size=math.floor(self.subsample*len(y)),
                                     replace=False)
