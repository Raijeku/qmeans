from turtle import settiltangle
from numpy import ones_like
import pytest
from qmeans.qkmeans import *
from hypothesis import given, assume, settings, example
from hypothesis.strategies import lists, integers, composite
from hypothesis.extra.numpy import arrays, array_shapes

#data_0 = np.array([[5,10]])
data_1 = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
x_1 = np.array([1,3,5,7,9])
y_1 = np.array([1,1,1,1,1])
x_2 = np.array([1,2])
y_2 = np.array([10,4])

def test_get_set_params_probability_random():
    qmeans = QuantumKMeans(max_iter=50, init='random', map_type='probability')
    params = qmeans.get_params()
    assert 'n_clusters' in params
    assert 'init' in params
    assert 'tol' in params
    assert 'verbose' in params
    assert 'max_iter' in params
    assert 'backend' in params
    assert 'map_type' in params
    assert 'shots' in params
    assert 'norm_relevance' in params
    assert 'initial_center' in params
    qmeans.set_params(**params)
    assert qmeans.get_params() == params

#works
#@pytest.fixture(scope='module')
#def qkmeans():
#    return QuantumKMeans(max_iter=2, init='random')

"""def test_preprocess_probability(data):
    assume(np.isfinite(data).all())
    #assume(data.any())
    preprocessed_data, norms = preprocess(data, map_type='probability')
    print('preprocessed')
    print(preprocessed_data)
    verification_data = data/((data**2).sum(axis=1)[:,np.newaxis]**(1/2))
    print(1)
    print(verification_data)
    verification_norms = (data**2).sum(axis=1)**(1/2)
    verification_data[np.isnan(verification_data)] = 0
    print(2)
    print(verification_data)
    verification_data[~np.isfinite(verification_data)] = 0
    print(3)
    print(verification_data)
    verification_norms[verification_norms == 0] = 1
    for i, point in enumerate(verification_data):
        if np.array_equiv(point, np.zeros_like(point)):
            #print('entered')
            point = np.ones_like(point)*((1/verification_data.shape[1])**(1/2))
            #print('new point is:')
            #print(point)
            verification_data[i] = point
    #if np.allclose(verification_data, np.zeros_like(verification_data)):
    #    verification_data = ones_like(verification_data)
    #if (verification_data == 0).all():
    #    verification_data = ones_like(verification_data)
    print('Test')
    print(preprocessed_data)
    print(verification_data)
    assert np.allclose(preprocessed_data, verification_data)
    print(norms)
    print(verification_norms)
    assert np.allclose(norms, verification_norms)"""

def test_preprocess_probability():
    data = data_1
    preprocessed_data, norms = preprocess(data, map_type='probability')
    verification_data = data/((data**2).sum(axis=1)[:,np.newaxis]**(1/2))
    verification_norms = (data**2).sum(axis=1)**(1/2)
    verification_data[np.isnan(verification_data)] = 0
    verification_data[~np.isfinite(verification_data)] = 0
    verification_norms[verification_norms == 0] = 1
    for i, point in enumerate(verification_data):
        if np.array_equiv(point, np.zeros_like(point)):
            #print('entered')
            point = np.ones_like(point)*((1/verification_data.shape[1])**(1/2))
            #print('new point is:')
            #print(point)
            verification_data[i] = point
    #if np.allclose(verification_data, np.zeros_like(verification_data)):
    #    verification_data = ones_like(verification_data)
    #if (verification_data == 0).all():
    #    verification_data = ones_like(verification_data)
    assert np.allclose(preprocessed_data, verification_data)
    assert np.allclose(norms, verification_norms)

#works
"""@given(arrays(np.float32,array_shapes(min_dims=2,max_dims=2,min_side=1,max_side=32)))
def test_preprocess_probability(data):
    assume(np.isfinite(data).all())
    data = data.astype('float64')
    #assume(data.any())
    preprocessed_data, norms = preprocess(data, map_type='probability')
    verification_data = data/((data**2).sum(axis=1)[:,np.newaxis]**(1/2))
    verification_norms = (data**2).sum(axis=1)**(1/2)
    verification_data[np.isnan(verification_data)] = 0
    verification_data[~np.isfinite(verification_data)] = 0
    verification_norms[verification_norms == 0] = 1
    for i, point in enumerate(verification_data):
        if np.array_equiv(point, np.zeros_like(point)):
            #print('entered')
            point = np.ones_like(point)*((1/verification_data.shape[1])**(1/2))
            #print('new point is:')
            #print(point)
            verification_data[i] = point
    #if np.allclose(verification_data, np.zeros_like(verification_data)):
    #    verification_data = ones_like(verification_data)
    #if (verification_data == 0).all():
    #    verification_data = ones_like(verification_data)
    assert np.allclose(preprocessed_data, verification_data)
    assert np.allclose(norms, verification_norms)
    """

def test_preprocess_angle():
    data = data_1
    preprocessed_data = preprocess(data, map_type='angle', norm_relevance=False)
    verification_norms = (data**2).sum(axis=1)**(1/2)
    verification_norms[verification_norms == 0] = 1
    if np.array_equiv(data, np.ones_like(data)*data[0]):
        verification_data = np.zeros_like(preprocessed_data)
    else:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        #std[std == 0] = 1
        verification_data = (data-mean)/std
        verification_data[np.isnan(verification_data)] = 0
    print("Data:")
    print(data)
    print("Preprocessed data:")
    print(preprocessed_data)
    print("Verification data:")
    print(verification_data)
    assert np.allclose(preprocessed_data, verification_data)
    
#works
"""
@given(arrays(np.float32,array_shapes(min_dims=2,max_dims=2,min_side=1,max_side=100)))
def test_preprocess_angle(data):
    assume(np.isfinite(data).all())
    data = data.astype('float64')
    preprocessed_data = preprocess(data, map_type='angle', norm_relevance=False)
    verification_norms = (data**2).sum(axis=1)**(1/2)
    verification_norms[verification_norms == 0] = 1
    if np.array_equiv(data, np.ones_like(data)*data[0]):
        verification_data = np.zeros_like(preprocessed_data)
    else:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        #std[std == 0] = 1
        verification_data = (data-mean)/std
        verification_data[np.isnan(verification_data)] = 0
    assert np.allclose(preprocessed_data, verification_data)
"""

"""def test_preprocess_angle(data):
    assume(np.isfinite(data).all())
    data = data.astype('float64')
    preprocessed_data = preprocess(data, map_type='angle', norm_relevance=False)
    if np.array_equiv(data, np.ones_like(data)*data[0]):
        verification_data = np.zeros_like(preprocessed_data)
    else:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        #std[std == 0] = 1
        verification_data = (data-mean)/std
        verification_data[np.isnan(verification_data)] = 0
    print(preprocessed_data)
    print(verification_data)
    assert np.allclose(preprocessed_data, verification_data)"""

#works
"""
@given(arrays(np.float32,array_shapes(min_dims=2,max_dims=2,min_side=1,max_side=100)))
#@example(data=np.array([[0., 0.]], dtype=np.float32))
#@example(data=np.array([[1.]], dtype=np.float32))
#@example(data=np.array([[0.]], dtype=np.float32))
@example(data=np.array([[0.], [1.]], dtype=np.float32))
def test_preprocess_angle_norm_relevance(data):
    assume(np.isfinite(data).all())
    data = data.astype('float64')
    preprocessed_data = preprocess(data, map_type='angle', norm_relevance=True)
    preprocessed_norms = preprocessed_data[:,-1:]
    preprocessed_data = preprocessed_data[:,:-1]
    print("data")
    print(data)
    verification_norms = (data**2).sum(axis=1)**(1/2)
    print("ver norm")
    print(verification_norms)
    #verification_norms[verification_norms == 0] = 1
    #print(verification_norms)
    max_norm = np.max(verification_norms)
    new_column = verification_norms/max_norm
    print(new_column)
    #print(preprocessed_norms)
    #new_column = new_column.reshape((new_column.size,1))
    verification_norms = np.reshape(new_column, preprocessed_norms.shape)
    #verification_norms = new_column[:,np.newaxis]
    #verification_norms = np.concatenate((np.empty_like(data), new_column),axis=1)
    #verification_norms[verification_norms == 0] = 1
    if np.array_equiv(data, np.ones_like(data)*data[0]):
        verification_data = np.zeros_like(preprocessed_data)
    else:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        #std[std == 0] = 1
        verification_data = (data-mean)/std
        verification_data[np.isnan(verification_data)] = 0

    verification_norms[np.isnan(verification_norms)] = 0

    print("Preprocessed data")
    print(preprocessed_data[:5])
    print("Verification data")
    print(verification_data[:5])
    print("Preprocessed norms")
    print(preprocessed_norms[:5])
    print("Verification norms")
    print(verification_norms[:5])
    assert np.allclose(preprocessed_data, verification_data)
    assert np.allclose(preprocessed_norms, verification_norms)
    """

def test_preprocess_angle_norm_relevance():
    data = data_1
    preprocessed_data = preprocess(data, map_type='angle', norm_relevance=True)
    preprocessed_norms = preprocessed_data[:,-1:]
    preprocessed_data = preprocessed_data[:,:-1]
    print("data")
    print(data)
    verification_norms = (data**2).sum(axis=1)**(1/2)
    print("ver norm")
    print(verification_norms)
    #verification_norms[verification_norms == 0] = 1
    #print(verification_norms)
    max_norm = np.max(verification_norms)
    new_column = verification_norms/max_norm
    print(new_column)
    #print(preprocessed_norms)
    #new_column = new_column.reshape((new_column.size,1))
    verification_norms = np.reshape(new_column, preprocessed_norms.shape)
    #verification_norms = new_column[:,np.newaxis]
    #verification_norms = np.concatenate((np.empty_like(data), new_column),axis=1)
    #verification_norms[verification_norms == 0] = 1
    if np.array_equiv(data, np.ones_like(data)*data[0]):
        verification_data = np.zeros_like(preprocessed_data)
    else:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        #std[std == 0] = 1
        verification_data = (data-mean)/std
        verification_data[np.isnan(verification_data)] = 0

    verification_norms[np.isnan(verification_norms)] = 0

    #print("Preprocessed data")
    #print(preprocessed_data[:5])
    #print("Verification data")
    #print(verification_data[:5])
    #print("Preprocessed norms")
    #print(preprocessed_norms[:5])
    #print("Verification norms")
    #print(verification_norms[:5])
    assert np.allclose(preprocessed_data, verification_data)
    assert np.allclose(preprocessed_norms, verification_norms)

"""@given(x=arrays(np.float64, integers(min_value=2, max_value=100)), y=arrays(np.float64, integers(min_value=2, max_value=100)))
@settings(deadline=None)
def test_distance_probability(x, y, qkmeans):
    assume(np.isfinite(x).all())
    assume(np.isfinite(y).all())
    print(qkmeans)
    x, x_norm = preprocess(x[np.newaxis], map_type='probability')
    y, y_norm = preprocess(y[np.newaxis], map_type='probability')
    point_distance = distance(x, y, qkmeans.backend, map_type='probability', norms=np.array([x_norm, y_norm]))
    assert np.isscalar(point_distance)
    assert point_distance >= 0"""

#works
"""
@composite
def point(draw):
    size = draw(integers(min_value=2, max_value=32))
    x = draw(arrays(np.float32, size))
    y = draw(arrays(np.float32, size))
    return (x, y)
    """

#works
"""
@given(x_y = point())
@settings(deadline=None)
def test_distance_probability(x_y, qkmeans):
    x = x_y[0].astype('float64')
    y = x_y[1].astype('float64')
    assume(np.isfinite(x).all())
    assume(np.isfinite(y).all())
    x, x_norm = preprocess(x.reshape(1,-1), map_type='probability')
    y, y_norm = preprocess(y.reshape(1,-1), map_type='probability')
    point_distance = distance(x[0], y[0], qkmeans.backend, map_type='probability', norms=np.array([x_norm[0], y_norm[0]]))
    assert np.isscalar(point_distance)
    assert point_distance >= 0
    """

def test_distance_probability():
    x = x_1
    y = y_1
    qkmeans = QuantumKMeans(max_iter=50, init='random', map_type='probability')
    x, x_norm = preprocess(x.reshape(1,-1), map_type='probability')
    y, y_norm = preprocess(y.reshape(1,-1), map_type='probability')
    point_distance = distance(x[0], y[0], qkmeans.backend, map_type='probability', norms=np.array([x_norm[0], y_norm[0]]))
    assert np.isscalar(point_distance)
    assert point_distance >= 0

def test_distance_angle():
    x = x_2
    y = y_2
    data = np.array([x, y])
    qkmeans = QuantumKMeans(max_iter=50, init='random', map_type='angle')
    preprocessed_data = preprocess(data, map_type='angle')
    x = preprocessed_data[0]
    y = preprocessed_data[1]
    point_distance = distance(x, y, qkmeans.backend, map_type='angle')
    assert np.isscalar(point_distance)
    assert point_distance >= 0

"""def test_distance_probability(x, y, qkmeans):
    assume(np.isfinite(x).all())
    assume(np.isfinite(y).all())
    print(qkmeans)
    x, x_norm = preprocess(x[np.newaxis], map_type='probability')
    print('first')
    y, y_norm = preprocess(y[np.newaxis], map_type='probability')
    print('second')
    point_distance = distance(x, y, qkmeans.backend, map_type='probability', norms=np.array([x_norm, y_norm]))
    print('third')
    assert np.isscalar(point_distance)
    assert point_distance >= 0"""

#works
"""
@given(data = arrays(np.float32,array_shapes(min_dims=2,max_dims=2,min_side=1,max_side=32)), n_clusters = integers(min_value=2, max_value=8))
@settings(deadline=None)
def test_fit(data, n_clusters):
    assume(np.isfinite(data).all())
    qkmeans = QuantumKMeans(max_iter=2, init='random', n_clusters=n_clusters)
    assume(data.shape[0] >= qkmeans.n_clusters)
    data = data.astype('float64')
    qkmeans.fit(data)
    assert qkmeans.labels_.size == data.shape[0]
    assert qkmeans.cluster_centers_.shape[0] <= qkmeans.n_clusters
    assert qkmeans.n_iter_ <= qkmeans.max_iter
    """

def test_fit_probability_random():
    data = data_1
    n_clusters = 3
    qkmeans = QuantumKMeans(max_iter=50, init='random', n_clusters=n_clusters, verbose = True)
    data = data.astype('float64')
    qkmeans.fit(data)
    assert qkmeans.cluster_centers_.shape[0] == n_clusters
    assert qkmeans.labels_.size == data.shape[0]
    assert qkmeans.cluster_centers_.shape[0] <= qkmeans.n_clusters
    assert qkmeans.n_iter_ <= qkmeans.max_iter

def test_fit_probability_qmeanspp():
    data = data_1
    n_clusters = 3
    qkmeans = QuantumKMeans(max_iter=50, init='qk-means++', n_clusters=n_clusters, verbose = True)
    data = data.astype('float64')
    qkmeans.fit(data)
    assert qkmeans.cluster_centers_.shape[0] == n_clusters
    assert qkmeans.labels_.size == data.shape[0]
    assert qkmeans.cluster_centers_.shape[0] <= qkmeans.n_clusters
    assert qkmeans.n_iter_ <= qkmeans.max_iter

def test_fit_probability_qmeanspp_far():
    data = data_1
    n_clusters = 3
    qkmeans = QuantumKMeans(max_iter=50, init='qk-means++', n_clusters=n_clusters, verbose = True, initial_center = 'far')
    data = data.astype('float64')
    qkmeans.fit(data)
    assert qkmeans.cluster_centers_.shape[0] == n_clusters
    assert qkmeans.labels_.size == data.shape[0]
    assert qkmeans.cluster_centers_.shape[0] <= qkmeans.n_clusters
    assert qkmeans.n_iter_ <= qkmeans.max_iter

def test_fit_angle_random():
    data = data_1
    n_clusters = 3
    qkmeans = QuantumKMeans(max_iter=50, init='random', n_clusters=n_clusters, verbose = True, map_type='angle')
    data = data.astype('float64')
    qkmeans.fit(data)
    assert qkmeans.cluster_centers_.shape[0] == n_clusters
    assert qkmeans.labels_.size == data.shape[0]
    assert qkmeans.cluster_centers_.shape[0] <= qkmeans.n_clusters
    assert qkmeans.n_iter_ <= qkmeans.max_iter

def test_fit_angle_qmeanspp():
    data = data_1
    n_clusters = 3
    qkmeans = QuantumKMeans(max_iter=50, init='qk-means++', n_clusters=n_clusters, verbose = True, map_type='angle')
    data = data.astype('float64')
    qkmeans.fit(data)
    assert qkmeans.cluster_centers_.shape[0] == n_clusters
    assert qkmeans.labels_.size == data.shape[0]
    assert qkmeans.cluster_centers_.shape[0] <= qkmeans.n_clusters
    assert qkmeans.n_iter_ <= qkmeans.max_iter

def test_fit_angle_qmeanspp_far():
    data = data_1
    n_clusters = 3
    qkmeans = QuantumKMeans(max_iter=50, init='qk-means++', n_clusters=n_clusters, verbose = True, map_type='angle', initial_center = 'far')
    data = data.astype('float64')
    qkmeans.fit(data)
    assert qkmeans.cluster_centers_.shape[0] == n_clusters
    assert qkmeans.labels_.size == data.shape[0]
    assert qkmeans.cluster_centers_.shape[0] <= qkmeans.n_clusters
    assert qkmeans.n_iter_ <= qkmeans.max_iter

#works
"""
@given(data = arrays(np.float32,array_shapes(min_dims=2,max_dims=2,min_side=1,max_side=32)), n_clusters = integers(min_value=2, max_value=8))
@settings(deadline=None)
def test_predict(data, n_clusters, qkmeans):
    assume(np.isfinite(data).all())
    qkmeans = QuantumKMeans(max_iter=2, init='random', n_clusters=n_clusters)
    assume(data.shape[0] >= qkmeans.n_clusters)
    data = data.astype('float64')
    qkmeans.fit(data)
    labels = qkmeans.predict(data)
    assert np.array_equiv(labels, qkmeans.labels_)
    """

def test_predict_probability_random():
    data = data_1
    n_clusters = 2
    qkmeans = QuantumKMeans(max_iter=50, init='random', n_clusters=n_clusters)
    data = data.astype('float64')
    qkmeans.fit(data)
    labels = qkmeans.predict(data)
    assert np.array_equiv(labels, qkmeans.labels_)

def test_predict_probability_qmeanspp():
    data = data_1
    n_clusters = 2
    qkmeans = QuantumKMeans(max_iter=50, init='qk-means++', n_clusters=n_clusters)
    data = data.astype('float64')
    qkmeans.fit(data)
    labels = qkmeans.predict(data)
    assert np.array_equiv(labels, qkmeans.labels_)

def test_predict_probability_qmeanspp_far():
    data = data_1
    n_clusters = 2
    qkmeans = QuantumKMeans(max_iter=50, init='qk-means++', n_clusters=n_clusters, initial_center = 'far')
    data = data.astype('float64')
    qkmeans.fit(data)
    labels = qkmeans.predict(data)
    assert np.array_equiv(labels, qkmeans.labels_)

def test_predict_angle_random():
    data = data_1
    n_clusters = 2
    qkmeans = QuantumKMeans(max_iter=50, init='random', n_clusters=n_clusters, map_type='angle')
    data = data.astype('float64')
    qkmeans.fit(data)
    labels = qkmeans.predict(data)
    assert np.array_equiv(labels, qkmeans.labels_)

def test_predict_angle_qmeanspp():
    data = data_1
    n_clusters = 2
    qkmeans = QuantumKMeans(max_iter=50, init='qk-means++', n_clusters=n_clusters, map_type='angle')
    data = data.astype('float64')
    qkmeans.fit(data)
    labels = qkmeans.predict(data)
    assert np.array_equiv(labels, qkmeans.labels_)

def test_predict_angle_qmeanspp_far():
    data = data_1
    n_clusters = 2
    qkmeans = QuantumKMeans(max_iter=50, init='qk-means++', n_clusters=n_clusters, map_type='angle', initial_center = 'far')
    data = data.astype('float64')
    qkmeans.fit(data)
    labels = qkmeans.predict(data)
    assert np.array_equiv(labels, qkmeans.labels_)