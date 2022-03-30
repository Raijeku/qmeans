from turtle import settiltangle
from numpy import ones_like
import pytest
from qmeans.qkmeans import *
from hypothesis import given, assume, settings, example
from hypothesis.strategies import lists, integers, composite
from hypothesis.extra.numpy import arrays, array_shapes

#data_0 = np.array([[5,10]])
#data_1 = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

@pytest.fixture(scope='module')
def qkmeans():
    return QuantumKMeans(max_iter=2, init='random')

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

@given(arrays(np.float32,array_shapes(min_dims=2,max_dims=2,min_side=1,max_side=32)))
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

"""@given(arrays(np.float64,array_shapes(min_dims=2,max_dims=2,min_side=1,max_side=100)))
def test_preprocess_angle(data):
    assume(np.isfinite(data).all())
    data = data.astype('float64')
    preprocessed_data = preprocess(data, map_type='angle', norm_relevance=False)
    if np.allclose(data, np.ones_like(data)*data[0]):
        verification_data = np.zeros_like(preprocessed_data)
    else:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        #std[std == 0] = 1
        verification_data = (data-mean)/std
        verification_data[np.isnan(verification_data)] = 0
    assert np.allclose(preprocessed_data, verification_data)"""

"""@given(arrays(np.float64,array_shapes(min_dims=2,max_dims=2,min_side=1,max_side=100)))
def test_preprocess_angle_norm_relevance(data):
    assume(np.isfinite(data).all())
    preprocessed_data = preprocess(data, map_type='angle', norm_relevance=True)
    if np.allclose(data, np.ones_like(data)*data[0]):
        verification_data = np.zeros_like(preprocessed_data)
    else:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        #std[std == 0] = 1
        verification_data = (data-mean)/std
        verification_data[np.isnan(verification_data)] = 0
    assert np.allclose(preprocessed_data, verification_data)"""

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

@composite
def point(draw):
    size = draw(integers(min_value=2, max_value=32))
    x = draw(arrays(np.float32, size))
    y = draw(arrays(np.float32, size))
    return (x, y)

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

@given(data = arrays(np.float32,array_shapes(min_dims=2,max_dims=2,min_side=1,max_side=32)), n_clusters = integers(min_value=2, max_value=8))
@settings(deadline=None)
def test_fit(data, n_clusters, qkmeans):
    assume(np.isfinite(data).all())
    assume(data.shape[0] >= qkmeans.n_clusters)
    qkmeans = QuantumKMeans(max_iter=2, init='random', n_clusters=n_clusters)
    data = data.astype('float64')
    qkmeans.fit(data)
    assert qkmeans.labels_.size == data.shape[0]
    assert qkmeans.cluster_centers_.shape[0] <= qkmeans.n_clusters
    assert qkmeans.n_iter_ <= qkmeans.max_iter

@given(data = arrays(np.float32,array_shapes(min_dims=2,max_dims=2,min_side=1,max_side=32)), n_clusters = integers(min_value=2, max_value=8))
@settings(deadline=None)
def test_predict(data, n_clusters, qkmeans):
    assume(np.isfinite(data).all())
    assume(data.shape[0] >= qkmeans.n_clusters)
    qkmeans = QuantumKMeans(max_iter=2, init='random', n_clusters=n_clusters)
    data = data.astype('float64')
    qkmeans.fit(data)
    labels = qkmeans.predict(data)
    assert np.array_equiv(labels, qkmeans.labels_)
