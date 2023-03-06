[![GitHub license](https://img.shields.io/github/license/Raijeku/qmeans)](https://github.com/Raijeku/qmeans/blob/main/LICENSE)
[![Linter](https://img.shields.io/badge/code%20style-pylint-orange)](https://github.com/PyCQA/pylint)
[![codecov](https://codecov.io/gh/Raijeku/qmeans/branch/main/graph/badge.svg?token=CC7BQ1P8T8)](https://codecov.io/gh/Raijeku/qmeans)

[![Clustering example](https://raw.githubusercontent.com/Raijeku/qmeans/main/q-means.png)](https://qmeans.readthedocs.io/)
<br>
Q-Means algorithm implementation using Qiskit compatible with Scikit-Learn.

The **q-means** leverages quantum computing to calculate distances for the centroid assignment part
of the k-means unsupervised learning algorithm. It shares the same general steps its classical
counterpart has, and is used alongside quantum simulators and quantum devices. This implementation
uses **Qiskit** and is compatible with the **scikit-learn** library in order to exploit the
capabilities for machine learning offered by scikit-learn. You can use the q-means in the same way
you would use the
[k-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and
many options are available for data encoding and job batching.

> :exclamation: Job batching is introduced as a way to speed up training time on quantum devices for larger datasets. [2]

## Installation

The project is publicly available in PyPI and can be installed using the following command:

`pip install qmeans`

## Usage

Example code for usage of the qmeans algorithm:

```python
import numpy as np
import pandas as pd
from qmeans.qkmeans import *

backend = Aer.get_backend("aer_simulator_statevector")
X = pd.DataFrame(np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]))
qk_means = QuantumKMeans(backend, n_clusters=2, verbose=True)
qk_means.fit(X)
print(qk_means.labels_)
```

## Requirements

Python 3.6+

Q-Means uses [Qiskit](https://github.com/Qiskit/qiskit) for quantum computations, [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) for data preparation and estimator formatting, and [Pandas](https://github.com/pandas-dev/pandas) and [Numpy](https://github.com/numpy/numpy) for data manipulation.

## Publications

> [1]: D. Quiroga, P. Date and R. Pooser, "Discriminating Quantum States with Quantum Machine Learning," 2021 IEEE International Conference on Quantum Computing and Engineering (QCE), 2021, pp. 481-482, doi: 10.1109/QCE52317.2021.00088.

> [2]: D. Quiroga, P. Date and R. Pooser, "Discriminating Quantum States with Quantum Machine Learning," 2021 International Conference on Rebooting Computing (ICRC), 2021, pp. 56-63, doi: 10.1109/ICRC53822.2021.00018.

> [3]: D. Quiroga, J. Botia, "Q-means clustering coherent noise tolerance analysis," International Congress EXPOIngenieria, 2022, pp. 437-443.
      
## License

This source code is free and open source, released under the Apache License, Version 2.0.