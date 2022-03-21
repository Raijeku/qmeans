import pytest
from qmeans.qkmeans import *

data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

def test_preprocess_probability():
    assert np.all(np.sum(data**2,axis=0) == np.ones(data.shape[1])*data.shape[0])