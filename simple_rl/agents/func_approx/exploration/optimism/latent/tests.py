from . import naive_spread_counter as nsc
from .CountingLatentSpaceClass import CountingLatentSpace
import numpy as np


class TestCounter:
    def setup_method(self, method):
        """ Pytest: setup object of class we want to test. """
        self.class_counter = CountingLatentSpace(0.1, "raw")

    def test_get_count_from_distances(self):
        distances = np.arange(0., 1., 0.1).reshape(2, 5)
        assert True

    def test_get_count_from_zero_distances(self):
        distances = np.zeros((2, 5))
        counts = self.class_counter._get_raw_count_from_distances(distances)
        assert counts.shape == (2,), counts.shape
        assert np.all(counts == 5)

    def test_get_count_from_inf_distances(self):
        distances = np.inf * np.ones((2, 5))
        counts = self.class_counter._get_raw_count_from_distances(distances)

        assert counts.shape == (2,), counts.shape
        assert np.all(counts == 0)

    def test_get_count_from_eps_distances(self):
        assert self.class_counter.epsilon == 0.1

        distances = np.asarray([[0.1]])
        counts = self.class_counter._get_raw_count_from_distances(distances)

        assert counts.shape == (1,), counts.shape
        assert np.all(counts == np.exp(-1))

        distances = np.inf * np.ones((2, 5))
        counts = self.class_counter._get_raw_count_from_distances(distances)

        assert counts.shape == (2,), counts.shape
        assert np.all(counts == 0)

    def test_get_raw_counts(self):
        assert self.class_counter.phi_type == "raw"

        buffer = np.arange(0., 1., 0.1).reshape((2, 5))
        states = np.copy(buffer)
        self.class_counter.train(buffer)
        counts = self.class_counter.get_counts(states)

        ans1 = np.exp(-(5 * (0.5**2)) / (self.class_counter.epsilon ** 2))
        ans2 = np.ones((2,)) + ans1

        assert counts.shape == (2,), counts.shape
        assert np.all(counts == ans2), counts


def test_differences_on_one_element_works():
    s_base = np.asarray([[1,2,3,4,5]])

    s1 = np.asarray([[1,2,3,4,5]])
    difference = nsc.get_all_distances_to_buffer(s_base,s1)
    assert difference.shape == (1,1)
    assert difference == 0, difference

    s2 = np.asarray([[1,2,3,4,6]])
    difference = nsc.get_all_distances_to_buffer(s_base,s2)
    assert difference.shape == (1,1)
    assert difference == 1, difference

    s3 = np.asarray([[1,2,3,3,6]])
    difference = nsc.get_all_distances_to_buffer(s_base,s3)
    assert difference.shape == (1,1)
    assert difference == np.sqrt(2), difference

    s4 = np.asarray([[1,2,3,4,7]])
    difference = nsc.get_all_distances_to_buffer(s_base, s4)
    assert difference.shape == (1,1)
    assert difference == np.sqrt(4), difference


def test_differences_on_two_elements_works():
    s_base = np.asarray([[1,2,3,4,5], [2,3,4,5,6]])

    s1 = np.asarray([[1,2,3,4,5], [2,3,4,5,6]])
    difference = nsc.get_all_distances_to_buffer(s_base, s1)
    assert np.all(difference == np.sqrt(np.asarray([[0,5],[5,0]])))

def test_differences_on_two_elements_with_different_sizes_works():
    s_base = np.asarray([[1,2,3,4,5], [2,3,4,5,6]])

    s1 = np.asarray([[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]])
    difference = nsc.get_all_distances_to_buffer(s_base, s1)
    assert np.all(difference == np.sqrt(np.asarray([[0,5,20],[5,0, 5]])))
