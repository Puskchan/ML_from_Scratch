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



class TreeBooster():
    def __init__(self, X, g, h, params, max_depth, idxs=None):
        # Initialize parameters
        self.params = params
        self.max_depth = max_depth
        assert self.max_depth >= 0, 'max_depth must be non-negative'

        # Set default values for params
        self.min_child_weight = params.get('min_child_weight', 1.0)
        self.reg_lambda = params.get('reg_lambda', 1.0)
        self.gamma = params.get('gamma', 0.0)
        self.colsample_bynode = params.get('colsample_bynode', 1.0)

        # Convert gradients and hessians from pd series to np arrays
        if isinstance(g,pd.Series): g = g.values
        if isinstance(h,pd.Series): h = h.values

        # Set the indices to use
        if idxs is None: idxs = np.arange(len(g))

        # Store feature, gradients, hessians and indices
        self.X, self.g, self.h, self.idxs = X, g, h, idxs
        self.n, self.c = len(idxs), X.shape[1]

        # Compute the initial value for the leaf node
        self.value = -g[idxs].sum() / (h[idxs].sum() + self.reg_lambda)
        self.best_score_so_far = 0.

        # If max_depth > 0, try to split and create child nodes
        if self.max_depth > 0:
            self._maybe_insert_child_nodes()

    
    def _maybe_insert_child_nodes(self):
        # Try to find the best split for each feature
        for i in range(self.c):
            self._find_better_split(i)
        

        # If this nde is leaf, stop
        if self.is_leaf: return

        # Split the data based on the best feature and threshold
        x = self.X.values[self.idxs, self.split_feature_idx]
        left_idx = np.nonzero(x <= self.threshold)[0]
        right_idx = np.nonzero(x > self.threshold)[0]
        
        # Create left and right child nodes
        self.left = TreeBooster(self.X, self.g, self.h, self.params, 
                                self.max_depth - 1, self.idxs[left_idx])
        self.right = TreeBooster(self.X, self.g, self.h, self.params, 
                                 self.max_depth - 1, self.idxs[right_idx])
        
        
    @property
    def is_leaf(self):
        # Check if this node is a leaf
        return self.best_score_so_far == 0.

    def _find_better_split(self, feature_idx):
        # Placeholder for method to find the best split
        pass