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

    @staticmethod
    def get_initiation_set_factors(matrix):
        assert matrix.shape[-1] == InitiationSet.state_size
        return matrix[..., InitiationSet.factor_indices]

    def predict(self, matrix):
        matrix = self.get_initiation_set_factors(matrix)
        return self.svm.predict(matrix)

    def decision_function(self, matrix):
        matrix = self.get_initiation_set_factors(matrix)
        return self.svm.decision_function(matrix)
        
    def fit(self, pos_matrix, neg_matrix=None):
        raise NotImplementedError("Define a fit function")

    def _init_svm(self, **kwargs):
        raise NotImplementedError("Define what type of svm to use")


class OneClassInitiationSet(InitiationSet):
    def __init__(self, **kwargs):
        InitiationSet.__init__(self, **kwargs)

    def fit(self, pos_matrix, neg_matrix=None):
        pos_matrix = self.get_initiation_set_factors(pos_matrix)
        self.svm.fit(pos_matrix)

    def _init_svm(self, **kwargs):
        return svm.OneClassSVM(**kwargs)


class TwoClassInitiationSet(InitiationSet):
    def __init__(self, **kwargs):
        InitiationSet.__init__(self, **kwargs)

    def fit(self, pos_matrix, neg_matrix=None):
        assert neg_matrix is not None
        pos_matrix = self.get_initiation_set_factors(pos_matrix)
        self.svm.fit(pos_matrix, neg_matrix)

    def _init_svm(self, **kwargs):
        return svm.SVC(**kwargs)
