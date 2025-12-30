import numpy as np
import time
import sklearn.metrics


def compute_quantiles(x, axis, levels):

    levels_clipped = np.clip(levels, 0., 1.)
    x_sorted = np.sort(x, axis)
    sample_size = x.shape[axis]
    idxs = (sample_size - 1) * levels_clipped
    idxs_low = np.floor(idxs).astype(int)
    idxs_high = np.ceil(idxs).astype(int)

    idxs_low = np.expand_dims(idxs_low, axis)
    idxs_high = np.expand_dims(idxs_high, axis)

    values_low = np.squeeze(np.take_along_axis(x_sorted, idxs_low, axis), axis)
    values_high = np.squeeze(np.take_along_axis(x_sorted, idxs_high, axis), axis)

    fractions = idxs - np.floor(idxs)

    quantiles = values_low + fractions * (values_high - values_low)

    quantiles = np.where(levels <= 0., 0., quantiles)
    quantiles = np.where(levels >= 1., np.inf, quantiles)

    return quantiles


def compute_pairwise_distances(x, y):
    """
    x: numpy array of shape (m_0, ..., m_{k-1}, feature dim.) where k >= 0
    y: numpy array of shape (n_0, ..., n_{l-1}, feature dim.) where l >= 0
    return: numpy array of shape (m_0, ..., m_{k-1}, n_0, ..., n_{l-1}) containing the pairwise distances between x & y
    """
    feature_dim = x.shape[-1]
    assert feature_dim == y.shape[-1]

    batch_shape_x = x.shape[:-1]
    batch_shape_y = y.shape[:-1]
    # flattened so that # dim = 2
    X = np.reshape(x, newshape=(-1, feature_dim))
    Y = np.reshape(y, newshape=(-1, feature_dim))
    # matrix of shape (m_0 x ... x m_{k-1}, n_0 x ... x n_{l-1})
    D = sklearn.metrics.pairwise_distances(X, Y, metric='euclidean')
    batch_shape = batch_shape_x + batch_shape_y
    D = np.reshape(D, newshape=batch_shape)
    return D



def compute_pairwise_distances_along_axis(x, y, axis):
    """
    x: numpy array of shape (m, feature dim.) where k >= 0 & m = (m_0, ..., m_{k-1}): multi-index
    y: numpy array of shape (n, feature dim.) where l >= 0 & n = (n_0, ..., n_{l-1}): multi-index
    axis : tuple of two integers i & j representing the axis of x & axis of y along which the computation is performed, respectively
           Note that i != k, j != l, and m_i must be equal to n_j.
    return: numpy array of shape (dim, m_{-i}, n_{-j}) containing the pairwise distances between x & y, where dim: dimension at the given axis
    """
    ndim_x = x.ndim
    ndim_y = y.ndim

    assert ndim_x >= 2 and ndim_y >= 2

    axis_x, axis_y = axis

    axis_dim = x.shape[axis_x]
    assert axis_dim == y.shape[axis_y]

    # axis != last axis
    assert axis_x % ndim_x != ndim_x - 1
    assert axis_y % ndim_y != ndim_y - 1

    # same feature dim
    feature_dim = x.shape[-1]
    assert feature_dim == y.shape[-1]

    batch_shape_x = x.shape[:-1]
    batch_shape_y = y.shape[:-1]

    n_batch_dim_x = len(batch_shape_x)
    n_batch_dim_y = len(batch_shape_y)

    # permute the axes
    X = np.moveaxis(x, axis_x, 0)
    Y = np.moveaxis(y, axis_y, 0)

    # flatten the arrays
    X = np.reshape(X, newshape=(axis_dim, -1, 1, feature_dim))
    Y = np.reshape(Y, newshape=(axis_dim, 1, -1, feature_dim))

    # new shape except the specified axis
    batch_shape_x = tuple(batch_shape_x[i] for i in range(n_batch_dim_x) if i != axis_x)
    batch_shape_y = tuple(batch_shape_y[i] for i in range(n_batch_dim_y) if i != axis_y)

    Ds = []
    for i in range(axis_dim):
        D = compute_pairwise_distances(X[i], Y[i])
        Ds.append(D)

    D_final = np.array(Ds)

    final_shape = (axis_dim,) + batch_shape_x + batch_shape_y
    D_final = np.reshape(D_final, newshape=final_shape)
    return D_final


def test_quantile_computation():
    x = np.random.rand(2, 1000, 2)
    levels = np.random.rand(2, 2)
    levels[0, 0] = -0.2
    levels[1, 1] = 1.4
    print('levels:', levels)
    quantiles = compute_quantiles(x, axis=1, levels=levels)
    print('quantiles:', quantiles)


def test_pairwise_distance_computation():
    batch_shape_x = (729,)
    batch_shape_y = (100, 20)

    n_batch_dim_x = len(batch_shape_x)
    n_batch_dim_y = len(batch_shape_y)
    feature_dim = 2

    shape_x = batch_shape_x + (feature_dim,)
    shape_y = batch_shape_y + (feature_dim,)

    x = np.random.rand(*shape_x)
    y = np.random.rand(*shape_y)

    # scikit-learn implementation
    begin = time.time()
    D1 = compute_pairwise_distances(x, y)
    print('scikit-learn: time={:.6f}sec'.format(time.time() - begin))

    # alternative computation using based on broadcasting
    begin = time.time()
    extra_axes = tuple(range(n_batch_dim_x, n_batch_dim_x+n_batch_dim_y))
    x2 = np.expand_dims(x, axis=extra_axes)

    D2 = np.sum((x2 - y) ** 2, axis=-1) ** .5
    print('broadcasting: time={:.6f}sec'.format(time.time()-begin))
    print('max. diff. between scikit-learn & broadcasting:', np.max(np.abs(D1 - D2)))


def test_pairwise_distance_computation_along_axis():
    """
    The loop version is faster in general; may benefit from the optimized scikit-learn implementation
    """
    batch_shape_x = (729, 12)
    batch_shape_y = (100, 20, 12)

    feature_dim = 2

    shape_x = batch_shape_x + (feature_dim,)
    shape_y = batch_shape_y + (feature_dim,)

    x = np.random.rand(*shape_x)
    y = np.random.rand(*shape_y)
    axis = (1, 2)

    begin = time.time()
    D1 = compute_pairwise_distances_along_axis(x, y, axis=axis)
    print('loop: time={:.6f}sec'.format(time.time() - begin))


if __name__ == "__main__":
    test_quantile_computation()