import numpy as np
from sklearn import svm
from simple_rl.mdp.StateClass import State

class InitiationSet(object):
    state_size = None
    factor_indices = None

    def __init__(self, **kwargs):
        assert self.state_size is not None
        assert self.factor_indices is not None

        self.svm = self._init_svm(**kwargs)

    def _get_relevant_factors(self, matrix):
        assert matrix.shape[-1] == self.state_size
        return matrix[..., self.factor_indices]

    def predict(self, matrix):
        matrix = self._get_relevant_factors(matrix)
        return self.svm.predict(matrix)

    def decision(self, matrix):
        matrix = self._get_relevant_factors(matrix)
        return self.svm.decision_function(matrix)
        
    def fit(self, pos_matrix, neg_matrix=None):
        raise NotImplementedError("Define a fit function")

    def _init_svm(self, **kwargs):
        raise NotImplementedError("Define what type of svm to use")


class OneClassInitiationSet(InitiationSet):

    def __init__(self, **kwargs):
        InitiationSet.__init__(**kwargs) 

    def fit(self, pos_matrix, neg_matrix=None):
        pos_matrix = self._get_relevant_factors(pos_matrix)
        self.svm.fit(pos_matrix)

    def _init_svm(self, **kwargs):
        return svm.OneClassSVM(**kwargs)


class TwoClassInitiationSet(InitiationSet):

    def __init__(self, **kwargs):
        InitiationSet.__init__(**kwargs) 

    def fit(self, pos_matrix, neg_matrix=None):
        assert neg_matrix is not None
        pos_matrix = self._get_relevant_factors(pos_matrix)
        neg_matrix =  self._get_relevant_factors(neg_matrix)
        self.svm.fit(pos_matrix, neg_matrix)

    def _init_svm(self, **kwargs):
        return svm.SVC(**kwargs)
