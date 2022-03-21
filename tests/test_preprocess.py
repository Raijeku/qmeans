import pytest
from qmeans.qkmeans import *

def test_preprocess_probability():
    data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    print(data)
    assert 1 == 1