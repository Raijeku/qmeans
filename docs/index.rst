.. qmeans documentation master file, created by
   sphinx-quickstart on Tue Apr 30 00:50:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Q-means documentation
==================================

The **q-means** leverages quantum computing to calculate distances for the centroid assignment part
of the k-means unsupervised learning algorithm. It shares the same general steps its classical
counterpart has, and is used alongside quantum simulators and quantum devices. This implementation
uses **Qiskit** and is compatible with the **scikit-learn** library in order to exploit the
capabilities for machine learning offered by scikit-learn. You can use the q-means in the same way
you would use the
`k-means <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_, and
many options are available for data encoding and job batching.

.. note::
   Job batching is introduced as a way to speed up training time on quantum devices for larger
   datasets. [2]_

Installation
-------------------

The project is publicly available in PyPI and can be installed using the following command:

.. parsed-literal::
   pip install qmeans

Requirements
-------------------

Q-Means uses `Qiskit <https://github.com/Qiskit/qiskit>`_ for quantum computations, 
`Scikit-Learn <https://github.com/scikit-learn/scikit-learn>`_ for data preparation and estimator formatting, 
and `Pandas <https://github.com/pandas-dev/pandas>`_ and `Numpy <https://github.com/numpy/numpy>`_ for data manipulation.

Publications
-------------------

.. [1] D. Quiroga, P. Date and R. Pooser, "Discriminating Quantum States with Quantum Machine Learning," 
   2021 IEEE International Conference on Quantum Computing and Engineering (QCE), 2021, pp. 481-482, doi: 10.1109/QCE52317.2021.00088.

.. [2] D. Quiroga, P. Date and R. Pooser, "Discriminating Quantum States with Quantum Machine Learning," 
   2021 International Conference on Rebooting Computing (ICRC), 2021, pp. 56-63, doi: 10.1109/ICRC53822.2021.00018.

.. [3] D. Quiroga, J. Botia, "Q-means clustering coherent noise tolerance analysis," 
   International Congress EXPOIngenieria, 2022, pp. 437-443.

.. toctree::
   :hidden:
   
   Home <self>
   api