"""Module for quantum k-means algorithm with a class containing sk-learn style functions resembling
the k-means algorithm.

This module contains the QuantumKMeans class for clustering according to euclidian distances
calculated by running quantum circuits.

Typical usage example::
    
    import numpy as np
    import pandas as pd
    from qkmeans import *

    backend = Aer.get_backend("aer_simulator_statevector")
    X = pd.DataFrame(np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]))
    qk_means = QuantumKMeans(backend, n_clusters=2, verbose=True, map_type='angle')
    qk_means.fit(X)
    print(qk_means.labels_)
"""
from typing import Tuple
import numpy as np
import pandas as pd
from qiskit import Aer, IBMQ, execute, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import IBMQBackend
from sklearn.preprocessing import normalize, scale
from sklearn.utils import check_random_state
from sklearn.utils. extmath import stable_cumsum

def preprocess(points: np.ndarray, map_type: str ='angle', norm_relevance: bool = False):
    """Preprocesses data points according to a type criteria.

    The algorithm scales the data points if the type is 'angle' and normalizes the data points
    if the type is 'probability'.

    Args:
        points: The input data points.
        map_type: {'angle', 'probability'} Specifies the type of data encoding.
            'angle': Uses U3 gates with its theta angle being the phase angle of the complex data
            point.
            'probability': Relies on data normalization to preprocess the data to acquire a norm of
            1.
        norm_relevance: If true, maps two-dimensional data onto 2 angles, one for the angle between
            both data points and another for the magnitude of the data points.

    Returns:
        p_points: Preprocessed points.
    """
    if map_type == 'angle':
        p_points = scale(points[:])
        a_points = points.copy()
        if norm_relevance is True:
            for i, point in enumerate(a_points):
                if np.array_equiv(point, np.zeros_like(point)):
                    point = np.ones_like(point)*((1/a_points.shape[1])**(1/2))
                a_points[i] = point
            _, norms = normalize(a_points[:], return_norm=True)
            #norms = np.sqrt(p_points[:,0]**2+p_points[:,1]**2)
            max_norm = np.max(norms)
            new_column = norms/max_norm
            new_column = new_column.reshape((new_column.size,1))
            p_points = np.concatenate((p_points, new_column),axis=1)
        return p_points
    elif map_type == 'probability':
        """if len(points.shape) > 1:
            size = points.shape[1]
        else:
            size = points.shape[0]"""
        for i, point in enumerate(points):
            if np.array_equiv(point, np.zeros_like(point)):
                point = np.ones_like(point)*((1/points.shape[1])**(1/2))
            points[i] = point
        p_points, norms = normalize(points[:], return_norm=True)
        return p_points, norms

def distance(x: np.ndarray, y: np.ndarray, backend: IBMQBackend, map_type: str = 'angle', shots: int = 1024, norms: np.ndarray = np.array([1, 1]), norm_relevance: bool = False):
    """Finds the distance between two data points by mapping the data points onto qubits using
    amplitude or angle encoding and then using a swap test.

    The algorithm performs angle encoding if the type is 'angle' and amplitude encoding if the type
    is 'probability'.

    Args:
        x: The first data point.
        y: The second data point.
        backend: IBM quantum device to calculate the distance with.
        map_type: {'angle', 'probability'} Specify the type of data encoding.
            'angle': Uses U3 gates with its theta angle being the phase angle of the complex data
            point.
            'probability': Relies on data normalization to preprocess the data to acquire a norm of
            1.
        shots: Number of repetitions of each circuit, for sampling.
        norm_relevance: If true, maps two-dimensional data onto 2 angles, one for the angle between
            both data points and another for the magnitude of the data points.

    Returns:
        distance: Distance between the two data points.
    """
    if map_type == 'angle':
        if x.size == 2:
            qubits = int(np.ceil(np.log2(x.size)))
            complexes_x = x[0] + 1j*x[1]
            complexes_y= y[0] + 1j*y[1]
            theta_1 = np.angle(complexes_x)
            theta_2 = np.angle(complexes_y)

            qr = QuantumRegister(3, name="qr")
            cr = ClassicalRegister(3, name="cr")

            qc = QuantumCircuit(qr, cr, name="k_means")
            qc.h(qr[0])
            qc.h(qr[1])
            qc.h(qr[2])
            qc.u3(theta_1, np.pi, np.pi, qr[1])
            qc.u3(theta_2, np.pi, np.pi, qr[2])
            qc.cswap(qr[0], qr[1], qr[2])
            qc.h(qr[0])

            qc.measure(qr[0], cr[0])
            qc.reset(qr)
            job = execute(qc,backend=backend, shots=shots)
            result = job.result()
            data = result.get_counts()
            if len(data)==1:
                return 0.0
            else: return data['0'*(qubits*2)+'1']/shots
        elif x.size == 3 and norm_relevance is True:
            qubits = int(np.ceil(np.log2(x.size)))
            complexes_x = x[0] + 1j*x[1]
            complexes_y= y[0] + 1j*y[1]
            theta_1 = np.angle(complexes_x)
            theta_2 = np.angle(complexes_y)

            ro_1 = x[2]*np.pi
            ro_2 = y[2]*np.pi

            qr = QuantumRegister(3, name="qr")
            cr = ClassicalRegister(3, name="cr")

            qc = QuantumCircuit(qr, cr, name="k_means")
            qc.h(qr[0])
            qc.h(qr[1])
            qc.h(qr[2])
            qc.u3(theta_1, np.pi, np.pi, qr[1])
            qc.u3(ro_1, 0, 0, qr[1])
            qc.u3(theta_2, np.pi, np.pi, qr[2])
            qc.u3(ro_2, 0, 0, qr[1])
            qc.cswap(qr[0], qr[1], qr[2])
            qc.h(qr[0])

            qc.measure(qr[0], cr[0])
            qc.reset(qr)
            job = execute(qc,backend=backend, shots=shots)
            result = job.result()
            data = result.get_counts()
            if len(data)==1: return 0.0
            else:
                return data['0'*(qubits*2)+'1']/shots
    elif map_type == 'probability':
        if x.size > 1:
            qubits = int(np.ceil(np.log2(x.size)))
        else:
            qubits = 1
        #print(y)
        n_x = np.zeros(2**qubits)
        n_x[:x.size] = x
        n_y = np.zeros(2**qubits)
        n_y[:y.size] = y
        qr = QuantumRegister(2*qubits + 1, name="qr")
        cr = ClassicalRegister(2*qubits + 1, name="cr")

        qc = QuantumCircuit(qr, cr, name="k_means")
        qc.initialize(n_x,[i+1 for i in range(qubits)])         # pylint: disable=no-member
        qc.initialize(n_y,[i+1+qubits for i in range(qubits)])  # pylint: disable=no-member

        qc.h(qr[0])
        for i in range(qubits):
            qc.cswap(qr[0], qr[1+i], qr[qubits+1+i])
        qc.h(qr[0])

        qc.measure(qr[0], cr[0])
        qc.reset(qr)
        job = execute(qc,backend=backend, shots=shots)
        result = job.result()
        data = result.get_counts()
        if len(data)==1:
            return 0.0
        else:
            M = data['0'*(qubits*2)+'1']/shots
            return (norms[0]**2 + norms[1] ** 2 - 2*norms[0]*norms[1]*((1 - 2*M)**(1/2)))**(1/2)

def batch_separate(X: np.ndarray, clusters: np.ndarray, max_experiments: int, norms: np.ndarray, cluster_norms: np.ndarray):
    """Creates batches of pairs of vectors.

    Separates data points X and cluster centers into a number of batches of elements for distance
    calculations in a single job. Each batch contains a set of data points and cluster centers,
    corresponding to the data for distance measurements in each batch.

    Args:
        X: Training instances to cluster.
        clusters: Cluster centers.
        max_experiments: The amount of distance measurements in each batch.

    Returns:
        B: Batches with pairs of data points and cluster centers.
    """
    if X.shape[0] > clusters.shape[0]:
        if X.shape[0] % max_experiments == 0:
            batches_X = np.asarray(np.split(X,[i*max_experiments for i in range(1,X.shape[0]//max_experiments)]))
            batches_norms_X = np.asarray(np.split(norms, [i*max_experiments for i in range(1, norms.shape[0]//max_experiments)]))
        else:
            batches_X = np.asarray(np.split(X,[i*max_experiments for i in range(1,X.shape[0]//max_experiments + 1)]))
            batches_norms_X = np.asarray(np.split(norms, [i*max_experiments for i in range(1, norms.shape[0]//max_experiments + 1)]))
        #print("batches_X:",batches_X)
        #print(batches_X.shape)
        #print("clusters:",clusters)
        #print(clusters.shape)
        if X.shape[0] % max_experiments == 0:
            batches_clusters = np.empty([(X.shape[0]//max_experiments)*clusters.shape[0],clusters.shape[1]], dtype=clusters.dtype)
            batches_norms_clusters = np.empty([(X.shape[0]//max_experiments)*cluster_norms.shape[0],1], dtype=cluster_norms.dtype)
        else:
            batches_clusters = np.empty([(X.shape[0]//max_experiments + 1)*clusters.shape[0],clusters.shape[1]], dtype=clusters.dtype)
            batches_norms_clusters = np.empty([(X.shape[0]//max_experiments + 1)*cluster_norms.shape[0],1], dtype=cluster_norms.dtype)
        for i in range(clusters.shape[0]):
            batches_clusters[i::clusters.shape[0]] = clusters[i]
            batches_norms_clusters[i::cluster_norms.shape[0]] = cluster_norms[i]
        #print("batches_clusters:",batches_clusters)
        #print(batches_clusters.shape)
        batches_X = np.asarray(np.repeat(batches_X,clusters.shape[0],axis=0))
        batches_norms_X = np.asarray(np.repeat(batches_norms_X,cluster_norms.shape[0],axis=0))
        #print("batches_X:",batches_X)
        #print(batches_X.shape)
        batches = ([(batches_X[i], batches_clusters[i]) for i in range(batches_clusters.shape[0])], [(batches_norms_X[i], batches_norms_clusters[i]) for i in range(batches_norms_clusters.shape[0])])
        return batches
    else:
        raise NotImplementedError

def batch_distance(B: Tuple[np.ndarray, np.ndarray], backend: IBMQBackend, norm_B: np.ndarray, map_type: str = 'angle', shots: int = 1024):
    """Finds the distance between pairs of data points and cluster centers inside a batch by
    mapping the data points onto qubits using amplitude or angle encoding and then using a swap test.

    The algorithm performs angle encoding if the type is 'angle' and amplitude encoding if the type
    is 'probability'.

    Args:
        B: The batch of X data points and y cluster centers.
        backend: IBM quantum device to calculate the distance with.
        map_type: {'angle', 'probability'} Specifies the type of data encoding.
            'angle': Uses U3 gates with its theta angle being the phase angle of the complex data
            point.
            'probability': Relies on data normalization to preprocess the data to acquire a norm of
            1.
        shots: Number of repetitions of each circuit, for sampling.

    Returns:
        distance: Distance between the data points and cluster centers of the batch.
    """
    if B[0].shape[1] == 2:
        if map_type == 'angle':
            qcs = []
            for point in B[0]:
                x = point
                y = B[1]
                complexes_x = x[0] + 1j*x[1]
                complexes_y= y[0] + 1j*y[1]
                theta_1 = np.angle(complexes_x)
                theta_2 = np.angle(complexes_y)

                qr = QuantumRegister(3, name="qr")
                cr = ClassicalRegister(3, name="cr")

                qc = QuantumCircuit(qr, cr, name="k_means")
                qc.h(qr[0])
                qc.h(qr[1])
                qc.h(qr[2])
                qc.u3(theta_1, np.pi, np.pi, qr[1])
                qc.u3(theta_2, np.pi, np.pi, qr[2])
                qc.cswap(qr[0], qr[1], qr[2])
                qc.h(qr[0])

                qc.measure(qr[0], cr[0])
                qc.reset(qr)
                qcs.append(qc)
            job = execute(qcs,backend=backend, shots=shots)
            result = job.result()
            data = result.get_counts()

            return [batch_data['001']/shots if len(batch_data)!=1 else 0.0 for batch_data in data]
        elif map_type == 'probability':
            qcs = []
            for point in B[0]:
                x = point
                y = B[1]
                qr = QuantumRegister(3, name="qr")
                cr = ClassicalRegister(3, name="cr")

                qc = QuantumCircuit(qr, cr, name="k_means")
                qc.initialize(x,1)  # pylint: disable=no-member
                qc.initialize(y,2)  # pylint: disable=no-member

                qc.h(qr[0])
                qc.cswap(qr[0], qr[1], qr[2])
                qc.h(qr[0])

                qc.measure(qr[0], cr[0])
                qc.reset(qr)
                qcs.append(qc)
            job = execute(qcs,backend=backend, shots=shots)
            result = job.result()
            data = result.get_counts()
            contained = ['0'*2+'1' in batch_data for batch_data in data]
            M = [data[i]['0'*2+'1']/shots if contained[i] is True else 0.0 for i in range(len(contained))]
            return [(norm_B[0][i]**2 + norm_B[1]**2 -2*norm_B[0][i]*norm_B[1]*((1 - 2*M_i)**(1/2)))**(1/2) for i, M_i in enumerate(M)]
    elif B[0].shape[1] == 3:
        if map_type == 'angle':
            qcs = []
            for point in B[0]:
                x = point
                y = B[1]
                complexes_x = x[0] + 1j*x[1]
                complexes_y= y[0] + 1j*y[1]
                theta_1 = np.angle(complexes_x)
                theta_2 = np.angle(complexes_y)

                ro_1 = x[2]*np.pi/2
                ro_2 = y[2]*np.pi/2

                qr = QuantumRegister(3, name="qr")
                cr = ClassicalRegister(3, name="cr")

                qc = QuantumCircuit(qr, cr, name="k_means")
                qc.h(qr[0])
                qc.h(qr[1])
                qc.h(qr[2])
                qc.u3(theta_1, np.pi, np.pi, qr[1])
                qc.u3(ro_1, 0, 0, qr[1])
                qc.u3(theta_2, np.pi, np.pi, qr[2])
                qc.u3(ro_2, 0, 0, qr[2])
                qc.cswap(qr[0], qr[1], qr[2])
                qc.h(qr[0])

                qc.measure(qr[0], cr[0])
                qc.reset(qr)
                qcs.append(qc)
            job = execute(qcs,backend=backend, shots=shots)
            result = job.result()
            data = result.get_counts()

            return [batch_data['001']/shots if len(batch_data)!=1 else 0.0 for batch_data in data]
    elif np.log2(B[0].shape[1]).is_integer():
        if map_type == 'angle':
            qcs = []
            for point in B[0]:
                x = point
                y = B[1]
                complexes_x = x[0] + 1j*x[1]
                complexes_y= y[0] + 1j*y[1]
                theta_1 = np.angle(complexes_x)
                theta_2 = np.angle(complexes_y)

                qr = QuantumRegister(3, name="qr")
                cr = ClassicalRegister(3, name="cr")

                qc = QuantumCircuit(qr, cr, name="k_means")
                qc.h(qr[0])
                qc.h(qr[1])
                qc.h(qr[2])
                qc.u3(theta_1, np.pi, np.pi, qr[1])
                qc.u3(theta_2, np.pi, np.pi, qr[2])
                qc.cswap(qr[0], qr[1], qr[2])
                qc.h(qr[0])

                qc.measure(qr[0], cr[0])
                qc.reset(qr)
                qcs.append(qc)
            job = execute(qcs,backend=backend, shots=shots)
            result = job.result()
            data = result.get_counts()

            return [batch_data['001']/shots if len(batch_data)!=1 else 0.0 for batch_data in data]
        elif map_type == 'probability':
            qcs = []
            for point in B[0]:
                x = point
                y = B[1]
                qr = QuantumRegister(int(np.log2(B[0].shape[1]))*2+1, name="qr")
                cr = ClassicalRegister(int(np.log2(B[0].shape[1]))*2+1, name="cr")

                qc = QuantumCircuit(qr, cr, name="k_means")
                qc.initialize(x,[i+1 for i in range(int(np.log2(B[0].shape[1])))])                              # pylint: disable=no-member
                qc.initialize(y,[i+1+int(np.log2(B[0].shape[1])) for i in range(int(np.log2(B[0].shape[1])))])  # pylint: disable=no-member

                qc.h(qr[0])
                for i in range(int(np.log2(B[0].shape[1]))):
                    qc.cswap(qr[0], qr[1+i], qr[int(np.log2(B[0].shape[1])+1)+i])
                qc.h(qr[0])

                qc.measure(qr[0], cr[0])
                qc.reset(qr)
                qcs.append(qc)
            job = execute(qcs,backend=backend, shots=shots)
            result = job.result()
            data = result.get_counts()
            contained = ['0'*int(np.log2(B[0].shape[1]))*2+'1' in batch_data for batch_data in data]
            M = [data[i]['0'*int(np.log2(B[0].shape[1]))*2+'1']/shots if contained[i] is True else 0.0 for i in range(len(contained))]
            #print('norm_B is', norm_B)
            #print('M is', M)
            return [(norm_B[0][i]**2 + norm_B[1]**2 -2*norm_B[0][i]*norm_B[1]*((1 - 2*M_i)**(1/2)))**(1/2) for i, M_i in enumerate(M)]
    else:
        if map_type == 'angle':
            qcs = []
            for point in B[0]:
                x = point
                y = B[1]
                complexes_x = x[0] + 1j*x[1]
                complexes_y= y[0] + 1j*y[1]
                theta_1 = np.angle(complexes_x)
                theta_2 = np.angle(complexes_y)

                qr = QuantumRegister(3, name="qr")
                cr = ClassicalRegister(3, name="cr")

                qc = QuantumCircuit(qr, cr, name="k_means")
                qc.h(qr[0])
                qc.h(qr[1])
                qc.h(qr[2])
                qc.u3(theta_1, np.pi, np.pi, qr[1])
                qc.u3(theta_2, np.pi, np.pi, qr[2])
                qc.cswap(qr[0], qr[1], qr[2])
                qc.h(qr[0])

                qc.measure(qr[0], cr[0])
                qc.reset(qr)
                qcs.append(qc)
            job = execute(qcs,backend=backend, shots=shots)
            result = job.result()
            data = result.get_counts()

            return [batch_data['001']/shots if len(batch_data)!=1 else 0.0 for batch_data in data]
        elif map_type == 'probability':
            qcs = []
            for point in B[0]:
                if np.log2(B[0].shape[1]).is_integer(): qubits = int(np.log2(B[0].shape[1]))
                else: qubits = int(np.log2(B[0].shape[1])) + 1
                x = np.zeros(2**qubits)
                x[:point.shape[0]] = point
                y = np.zeros(2**qubits)
                y[:B[1].shape[0]] = B[1]
                qr = QuantumRegister(qubits*2+1, name="qr")
                cr = ClassicalRegister(qubits*2+1, name="cr")

                qc = QuantumCircuit(qr, cr, name="k_means")
                qc.initialize(x,[i+1 for i in range(qubits)])           # pylint: disable=no-member
                qc.initialize(y,[i+1+qubits for i in range(qubits)])    # pylint: disable=no-member

                qc.h(qr[0])
                for i in qubits:
                    qc.cswap(qr[0], qr[1+i], qr[qubits+1+i])
                qc.h(qr[0])

                qc.measure(qr[0], cr[0])
                qc.reset(qr)
                qcs.append(qc)
            job = execute(qcs,backend=backend, shots=shots)
            result = job.result()
            data = result.get_counts()
            contained = ['0'*qubits*2+'1' in batch_data for batch_data in data]
            M = [data[i]['0'*qubits*2+'1']/shots if contained[i]==True else 0.0 for i in range(len(contained))]
            return [(norm_B[0][i]**2 + norm_B[1]**2 -2*norm_B[0][i]*norm_B[1]*((1 - 2*M_i)**(1/2)))**(1/2) for i, M_i in enumerate(M)]

def batch_collect(batch_d: np.ndarray, desired_shape: Tuple[int, int]):
    """Collects batches of distances.

    Retrieves batches of distances and transforms the shape of the data to a desired shape.

    Args:
        batch_d: Batches of distances.
        desired_shape: The shape of the collected distances.

    Returns:
        final_batch_d: Transformed distances.
    """
    #print('Batch d is', batch_d)
    #print('Batch d shape is', batch_d.shape)
    #print('Desired shape is', desired_shape)
    final_batch_d = np.empty(batch_d.shape, dtype=batch_d.dtype)
    #print('Final Batch D is', final_batch_d)
    for i in range(batch_d.shape[0]//desired_shape[0]):
        final_batch_d[i] = batch_d[desired_shape[0]*i]
    #print('Final Batch D is', final_batch_d)
    #print(batch_d.shape[0]//desired_shape[0], batch_d.shape[0])
    #print(batch_d.shape, desired_shape)
    if batch_d.squeeze(axis=-1).shape != desired_shape:
        for i in range(batch_d.shape[0]//desired_shape[0],batch_d.shape[0]):
            final_batch_d[i] = batch_d[desired_shape[0]*i-batch_d.shape[0]+1]
    #print('Final Batch D is', final_batch_d)
    return final_batch_d.reshape(desired_shape)

def batch_distances(X: np.ndarray, cluster_centers: np.ndarray, backend: IBMQBackend, map_type: str, shots: int, verbose: bool, norms: np.ndarray, cluster_norms: np.ndarray):
    """Batches data and calculates and collects distances.

    Data is separated into batches, sent to the quantum device to calculate distances and the
    distances are then collected from the results.

    Args:
        X: Training instances to cluster.
        cluster_centers: Coordinates of cluster centers.
        backend: IBM quantum device to run the quantum k-means algorithm on.
        map_type: {'angle', 'probability'} Specifies the type of data encoding.
            'angle': Uses U3 gates with its theta angle being the phase angle of the complex data
            point.
            'probability': Relies on data normalization to preprocess the data to acquire a norm of
            1.
        shots: Number of repetitions of each circuit, for sampling.
        verbose: Defines if verbosity is active for deeper insight into the class processes.

    Returns:
        distance: Distance between the data points and cluster centers.
    """
    #print('LOOK HERE OMG ----------------------------------------------------------------')
    #print(norms)
    #print('LOOK HERE CLUSTERS ----------------------------------------------------------------')
    #print(cluster_norms)
    #print('LOOK HERE END ----------------------------------------------------------------')

    if isinstance(cluster_centers, pd.DataFrame):
        batches, norm_batches = batch_separate(X.to_numpy(), cluster_centers.to_numpy(),backend.configuration().max_experiments, norms, cluster_norms)
    else: batches, norm_batches = batch_separate(X.to_numpy(), cluster_centers,backend.configuration().max_experiments, norms, cluster_norms)
    #if verbose: print('Batches are', batches)
    #if verbose: print('Norm atches are', norm_batches)
    distance_list = np.asarray([batch_distance(B,backend,norm_batches[i],map_type,shots) for i, B in enumerate(batches)])
    #if verbose: print('Distance list is', distance_list)
    distances = batch_collect(distance_list, (cluster_centers.shape[0],X.shape[0]))
    #if verbose: print('Distances are', distances)
    return distances

def qkmeans_plusplus(X: np.ndarray, n_clusters: int, backend: IBMQBackend, map_type: str, verbose: bool, initial_center: str, shots: int = 1024, norms: np.ndarray = np.array([1,1]), batch: bool = True, x_squared_norms: np.ndarray = None, n_local_trials: int = None, random_state: int = None):
    """Init n_clusters seeds according to qk-means++.

    Selects initial cluster centers for qk-mean clustering in a smart way to speed up convergence.

    Args:
        X: The data to pick seeds from.
        n_clusters: The number of centroids to initialize.
        backend: IBM quantum device to run the quantum k-means algorithm on.
        map_type: {'angle', 'probability'} Specifies the type of data encoding.
            'angle': Uses U3 gates with its theta angle being the phase angle of the complex data
            point.
            'probability': Relies on data normalization to preprocess the data to acquire a norm of
            1.
        verbose: Defines if verbosity is active for deeper insight into the class processes.
        initial_center: {'random', 'far'} Speficies the strategy for setting the initial cluster
        center.
            'random': Assigns a random initial center.
            'far': Specifies the furthest point as the initial center.
        x_squared_norms: Squared Euclidean norm of each data point.
        n_local_trials: The number of seeding trials for each center (except the first), of which
            the one reducing inertia the most is greedily chosen. Set to None to make the number of
            trials depend logarithmically on the number of seeds (2+log(k)).
            random_state: Determines random number generation for centroid initialization. Pass an int
            for reproducible output across multiple function calls.

    Returns:
        centers: The initial centers for qk-means.
        indices: The index location of the chosen centers in the data array X. For a given index
            and center, X[index] = center.
    """
    if verbose:
        print('Started Qkmeans++')
    random_state = check_random_state(random_state)
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype= X.values.dtype)

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    indices[0] = center_id
    centers[0] = X.values[center_id]

    if verbose:
        print('Centers are:', pd.DataFrame(centers))

    if batch:
        closest_distances = batch_distances(X, centers[0, np.newaxis], backend, map_type, shots, verbose)
    else: closest_distances = np.asarray([[distance(point,centroid,backend,map_type,shots,norms[i,j]) for i, point in X.iterrows()] for j, centroid in pd.DataFrame(centers[0, np.newaxis]).iterrows()])
    current_pot = closest_distances.sum()

    #if verbose:
    #    print('Closest distances are:', closest_distances)

    for c in range(1, n_clusters):
        if verbose:
            print('Cluster center', c)
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_distances), rand_vals)

        np.clip(candidate_ids, None, closest_distances.size - 1, out=candidate_ids)

        if batch:
            distance_to_candidates = batch_distances(X, X.values[candidate_ids], backend, map_type, shots, verbose)
        else: distance_to_candidates = np.asarray([[distance(point,centroid,backend,map_type,shots,norms[i,j]) for i, point in X.iterrows()] for j, centroid in X.iloc[candidate_ids].iterrows()])

        np.minimum(closest_distances, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_distances = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X.values[best_candidate]
        indices[c] = best_candidate

        if verbose:
            print('Centers are:', pd.DataFrame(centers))
        #if verbose: print('Closest distances are:', closest_distances)

        if c == 1 and initial_center == 'far':
            if batch:
                closest_distances = batch_distances(X, centers[1, np.newaxis], backend, map_type, shots, verbose)
            else: closest_distances = np.asarray([[distance(point,centroid,backend,map_type,shots,norms[i,j]) for i, point in X.iterrows()] for j, centroid in pd.DataFrame(centers[1, np.newaxis]).iterrows()])
            current_pot = closest_distances.sum()
            rand_vals = random_state.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(stable_cumsum(closest_distances), rand_vals)

            np.clip(candidate_ids, None, closest_distances.size - 1, out=candidate_ids)

            if batch:
                distance_to_candidates = batch_distances(X, X.values[candidate_ids], backend, map_type, shots, verbose)
            else: distance_to_candidates = np.asarray([[distance(point,centroid,backend,map_type,shots,norms[i,j]) for i, point in X.iterrows()] for j, centroid in X.iloc[candidate_ids].iterrows()])

            np.minimum(closest_distances, distance_to_candidates,
                    out=distance_to_candidates)
            candidates_pot = distance_to_candidates.sum(axis=1)

            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_distances = distance_to_candidates[best_candidate]
            best_candidate = candidate_ids[best_candidate]

            centers[0] = X.values[best_candidate]
            indices[0] = best_candidate

            if verbose:
                print('Centers are:', pd.DataFrame(centers))

    return centers, indices

class QuantumKMeans():
    """Quantum k-means clustering algorithm. This k-means alternative implements quantum machine
    learning to calculate distances between data points and centroids using quantum circuits.

    Args:
        n_clusters: The number of clusters to use and the amount of centroids generated.
        init: {'qk-means++, 'random'}, callable or array-like of shape (n_clusters, n_features)
        Method for initialization:
            'qk-means++' : selects initial cluster centers for qk-mean clustering in a smart way to
            speed up convergence.
            'random': choose n_clusters observations (rows) at random from data for the initial
            centroids.
            If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial
            centers.
            If a callable is passed, it should take arguments X, n_clusters and a random state and
            return an initialization.
        tol: Relative tolerance with regards to Frobenius norm of the difference in the cluster
            centers of two consecutive iterations to declare convergence.
        verbose: Defines if verbosity is active for deeper insight into the class processes.
        max_iter: Maximum number of iterations of the quantum k-means algorithm for a single run.
        backend: IBM quantum device to run the quantum k-means algorithm on.
        map_type: {'angle', 'probability'} Specifies the type of data encoding.
            'angle': Uses U3 gates with its theta angle being the phase angle of the complex data
            point.
            'probability': Relies on data normalization to preprocess the data to acquire a norm of
            1.
        shots: Number of repetitions of each circuit, for sampling.
        norm_relevance: If true, maps two-dimensional data onto 2 angles, one for the angle between
            both data points and another for the magnitude of the data points.
        initial_center: {'random', 'far'} Speficies the strategy for setting the initial cluster
            center.
            'random': Assigns a random initial center.
            'far': Specifies the furthest point as the initial center.

    Attributes:
        cluster_centers_: Coordinates of cluster centers.
        labels_: Centroid labels for each data point.
        n_iter_: Number of iterations run before convergence.
    """
    def __init__(self, backend: IBMQBackend = Aer.get_backend("aer_simulator_statevector"), n_clusters: int = 2, init: str = 'qk-means++', tol: float = 0.0001, max_iter: int = 300, verbose: bool = False, map_type: str = 'probability', shots: int = 1024, norm_relevance: bool = False, initial_center: str = 'random'):
        """Initializes an instance of the quantum k-means algorithm."""
        self.cluster_centers_ = np.empty(0)
        self.labels_ = np.empty(0)
        self.n_iter_ = 0
        self.n_clusters = n_clusters
        self.init = init
        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.backend = backend
        self.map_type = map_type
        self.shots = shots
        self.norm_relevance = norm_relevance
        self.initial_center = initial_center

    def fit(self, X: np.ndarray, batch: bool = False):
        """Computes quantum k-means clustering.

        Args:
            X: Training instances to cluster.
            batch: Option for using batches to calculate distances.

        Returns:
            self: Fitted estimator.
        """
        if self.verbose:
            print('Data is:',X)
        finished = False
        old_X = pd.DataFrame(X)
        if self.map_type == 'probability':
            X, norms = preprocess(X, self.map_type, self.norm_relevance)
            X = pd.DataFrame(X)
        else: X = pd.DataFrame(preprocess(X, self.map_type, self.norm_relevance))
        #print('Preprocessed data is:',X)
        if self.init == 'qk-means++':
            self.cluster_centers_, _ = qkmeans_plusplus(X, self.n_clusters, self.backend, self.map_type, self.verbose, self.initial_center, shots=self.shots, batch=batch, norms=norms)
            self.cluster_centers_ = pd.DataFrame(self.cluster_centers_).values
        elif self.init == 'random':
            self.cluster_centers_ = old_X.sample(n=self.n_clusters)
        #print('Cluster centers are:', self.cluster_centers_)
        iteration = 0
        while not finished and iteration<self.max_iter:
            if self.verbose:
                print("Iteration",iteration)
            normalized_clusters, cluster_norms = preprocess(self.cluster_centers_.values, self.map_type, self.norm_relevance)
            normalized_clusters = pd.DataFrame(normalized_clusters)
            #print(norms)
            #print(cluster_norms)
            #print(X, normalized_clusters)
            if batch:
                distances = batch_distances(X, normalized_clusters, self.backend, self.map_type, self.shots, self.verbose, norms, cluster_norms)
            else: distances = np.asarray([[distance(point,centroid,self.backend,self.map_type,self.shots,np.array([norms[i],cluster_norms[j]])) for i, point in X.iterrows()] for j, centroid in normalized_clusters.iterrows()])
            self.labels_ = np.asarray([np.argmin(distances[:,i]) for i in range(distances.shape[1])])
            print('self labels', self.labels_)
            new_centroids = old_X.groupby(self.labels_).mean() #Needs to be checked to see if less centers are an option
            print('new centroids', new_centroids)
            if self.verbose:
                print("Old centroids are",self.cluster_centers_)
            if self.verbose:
                print("New centroids are",new_centroids)
            if abs((new_centroids - self.cluster_centers_).sum(axis=0).sum()) < self.tol:
                finished = True
            self.cluster_centers_ = new_centroids
            if self.verbose:
                print("Centers are", self.labels_)
            self.n_iter_ += 1
            iteration += 1
        return self

    def predict(self, X: np.ndarray, sample_weight: np.ndarray = None, batch: bool = False):
        """Predict the closest cluster each sample in X belongs to.

        Args:
            X: New data points to predict.
            sample_weight: The weights for each observation in X. If None, all observations are
            assigned equal weight.
            batch: Option for using batches to calculate distances.

        Returns:
            labels: Centroid labels for each data point.
        """
        X, norms = pd.DataFrame(preprocess(X, self.map_type, self.norm_relevance))
        if sample_weight is None:
            if batch:
                distances = batch_distances(X, self.cluster_centers_, self.backend, self.map_type, self.shots, self.verbose)
            else: distances = np.asarray([[distance(point,centroid,self.backend,self.map_type,self.shots,norms[i,j]) for i,point in X.iterrows()] for j,centroid in self.cluster_centers_.iterrows()])
        else:
            weight_X = X * sample_weight
            if batch:
                batch_distances(weight_X, self.cluster_centers_, self.backend, self.map_type, self.shots, self.verbose)
            else: distances = np.asarray([[distance(point,centroid,self.backend,self.map_type,self.shots) for _,point in weight_X.iterrows()] for _,centroid in self.cluster_centers_.iterrows()])
        labels = np.asarray([np.argmin(distances[:,i]) for i in range(distances.shape[1])])
        return labels

    def get_params(self, deep: bool = True):
        """Get parameters for this estimator.

        Args:
            deep: If True, will return the parameters for this estimator and contained subobjects
                that are estimators.

        Returns:
            params: Parameter names mapped to their values.
        """
        return {"n_clusters": self.n_clusters, "init": self.init, "tol": self.tol, "verbose": self.verbose, "max_iter": self.max_iter, "backend": self.backend, "map_type": self.map_type, "shots": self.shots, "norm_relevance": self.norm_relevance, "initial_center": self.initial_center }

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
