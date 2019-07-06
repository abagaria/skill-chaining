# Python imports
import numpy as np
import pdb

''' StateClass.py: Contains the State Class. '''

class PortableState(object):
    ''' Abstract class for states defined over problem-space and agent-space. '''

    def __init__(self, pspace_data=[], aspace_data=[], pspace_is_terminal=False):
        self.pspace_data = pspace_data
        self.aspace_data = aspace_data
        self.data = self.get_data()
        self._is_terminal = pspace_is_terminal

    def pspace_features(self):
        '''
        Summary
            Used by function approximators to represent the state.
            Override this method in State subclasses to have function
            approximators use a different set of features.
        Returns:
            (iterable)
        '''
        return np.array(self.pspace_data).flatten()

    def aspace_features(self):
        return np.array(self.aspace_data).flatten()

    def combined_features(self):
        pspace = self.pspace_features()
        aspace = self.aspace_features()
        return np.concatenate((pspace, aspace), axis=0)

    def get_data(self):
        combined = self.combined_features()
        return combined.tolist()

    def get_num_feats(self):
        return len(self.pspace_features()) + len(self.aspace_features())

    def is_terminal(self):
        return self._is_terminal

    def set_terminal(self, is_term=True):
        self._is_terminal = is_term

    def initiation_classifier_features(self):
        pass

    @staticmethod
    def initiation_classifier_feature_indices():
        pass

    def __hash__(self):
        if type(self.data).__module__ == np.__name__:
            # Numpy arrays
            return hash(str(self.data))
        elif self.data.__hash__ is None:
            return hash(tuple(self.data))
        else:
            return hash(self.data)

    def __str__(self):
        return "s." + str(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
