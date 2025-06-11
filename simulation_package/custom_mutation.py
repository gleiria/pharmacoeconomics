import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.operators.mutation.pm import PolynomialMutation


class CustomMutation(Mutation):
    def __init__(self, prob =0.5):
        super().__init__()
        self.prob = float(prob)

    def _do(self, problem, X, **kwargs):
        """mutation logic is implemented here """
    
        for i, _ in enumerate(X):
            if np.random.rand() < self.prob:
                non_zero = np.where(X[i] != 0)[0]
                if len(non_zero) >0:
                    closest_to_zero = non_zero[np.argmin(np.abs(X[i][non_zero]))]
                    X[i][closest_to_zero] = 0
        return X
        

     
    

class CombinedMutation(Mutation):
    """combine two mutation to be passed to the algorithm"""
    
    def __init__(self, custom_mutation, default_mutation):
        super().__init__()
        self.custom_mutation = custom_mutation
        self.default_mutation = default_mutation

    def _do(self, problem, X, **kwargs):
        """h"""
        # default mutation
        X = self.default_mutation._do(problem, X, **kwargs)
        # custom mutation
        X = self.custom_mutation._do(problem, X, **kwargs)
        return X

                    


