from qmeans.qmeans import *
import pytest
from qiskit_ibm_runtime import QiskitRuntimeService

data_1 = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
data_2 = np.array([[1, 5], [1, 10], [1, -3]])
data_3 = np.array([[1, 2, 5], [1, 4, 10], [1, 0, -3], [10, 2, 8], [10, 4, 1], [10, 0, 20]])
data_4 = np.array([[1, 10, 5], [1, 20, 10], [1, 15, -3]])
data_5 = np.array([[1, 2, 5, 10], [1, 4, 10, 10], [1, 0, -3, 10], [10, 2, 8, 10], [10, 4, 1, 10], [10, 0, 20, 10]])
data_6 = np.array([[1, 10, 5, 10], [1, 20, 10, 10], [1, 15, -3, 10]])
x_1 = np.array([1,3,5,7,9])
y_1 = np.array([1,1,1,1,1])
x_2 = np.array([1,2])
y_2 = np.array([10,4])
x_3 = np.array([14,10,5])
y_3 = np.array([10,20,2])
x_4 = np.array([1,3,5,7,9,11])

#def test_wut(qiskit_token):
#    print(qiskit_token)
#    assert False

@pytest.fixture(scope='session')
def setup_qiskit(qiskit_token):
    QiskitRuntimeService.save_account(token=qiskit_token)

def test_batch_separate():
    X = data_1
    cluster_centers = data_2
    norms = x_4
    cluster_norms = x_3
    max_experiments = 4
    batches, norm_batches = batch_separate(X, cluster_centers, max_experiments, norms, cluster_norms)
    #print(batches)
    #print(norm_batches)
    assert len(batches) == len(norm_batches)
    assert len(batches[0]) == 2
    assert len(norm_batches[0]) == 2
    assert len(batches) == np.ceil(len(X)/max_experiments) * len(cluster_centers)

def test_batch_distance_probability():
    X = data_5
    cluster_centers = data_6
    X, norms = preprocess(X, 'probability')
    cluster_centers, cluster_norms = preprocess(cluster_centers, 'probability')
    max_experiments = 4
    provider = QiskitRuntimeService()
    backend = provider.get_backend('ibmq_qasm_simulator')
    qkmeans = QuantumKMeans(backend = backend, max_iter=50, init='random', n_clusters=len(cluster_centers), verbose = True, map_type='probability')
    batches, norm_batches = batch_separate(X, cluster_centers, max_experiments, norms, cluster_norms)
    #print(batches)
    #print(norm_batches)
    distance_list = np.asarray([batch_distance(B,qkmeans.backend,norm_batches[i],'probability',8096) for i, B in enumerate(batches)])
    #print(distance_list)
    assert len(batches) == len(distance_list)
    distances = 0
    for i in range(len(batches)):
        assert len(batches[i][0]) == len(distance_list[i])
        distances += len(distance_list[i])
    assert len(X) * len(cluster_centers) == distances

