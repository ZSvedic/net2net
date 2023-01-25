''' Data utils. '''

from math import log10
import numpy as np

####################################################################################################

def create_spiral_datasets(degrees):
    ''' Create and transform data. '''
    spirals = two_spirals(degrees, degrees) # Get two spirals.
    return NNDataSets(2, 1, ("A", "B"), spirals) # Inputs are (x,y), output is binary.

####################################################################################################

class NNDataSets:
    ''' Sets training data based on input dimension, output dimension, class names, and data. '''

    def __init__(self, in_dim, out_dim, class_names, data):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.class_names = class_names
        self.train, self.test, self.validate = \
            split_to_sets(
                *unison_shuffle(
                    *join(data)))

####################################################################################################

def two_spirals(total_points, degrees, noise=.5):
    ''' Returns an array of two arrays of tuples, each representing a spiral. Unmodified code:
    https://glowingpython.blogspot.com/2017/04/solving-two-spirals-problem-with-keras.html '''

    n_points = total_points//2
    radians = np.sqrt(np.random.rand(n_points, 1)) * degrees * (2*np.pi)/360
    d1x = -np.cos(radians)*radians + np.random.rand(n_points, 1) * noise
    d1y = np.sin(radians)*radians + np.random.rand(n_points, 1) * noise

    return [np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))]

####################################################################################################

def test_join():
    a = [[1, 1], [2, 2]]
    b = [[3, 3], [4, 4]]
    ab, col = join([a, b])
    ex_ab = [[1, 1], [2, 2], [3, 3], [4, 4]]
    ex_col = [0, 0, 1, 1]
    assert (ab == ex_ab).all() and (col == ex_col).all()

def join(data_arrays):
    ''' Returns data in a single array, and a corresponding classes array. '''
    return np.vstack(data_arrays), np.hstack(
        [np.full(len(a), i) for i, a in enumerate(data_arrays)])

####################################################################################################

def test_unison_shuffle():
    a, b = unison_shuffle(np.array([1, 2, 3]), np.array([10, 20, 30]))
    assert len(a) == 3 and len(b) == 3 and (a*10 == b).all()

def unison_shuffle(a, b):
    ''' Copied from: https://stackoverflow.com/a/4602224/10546849 '''
    assert len(a) == len(b) and isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    p = np.random.permutation(len(a))
    return (a[p], b[p])

####################################################################################################

def test_split_to_sets():
    def check(ds, size, inputs, labels):
        assert len(ds.inputs) == size and len(ds.labels) == size and \
            (ds.inputs == inputs).all() and (ds.labels == labels).all()
    d = np.array([1, -2, 3, -4, 5, -6, 7, -8])
    c = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    train, test, val = split_to_sets(d, c, splits=(0.5, 0.25, 0.25))
    check(train, 4, [1, -2, 3, -4], [0, 1, 0, 1])
    check(test, 2, [5, -6], [0, 1])
    check(val, 2, [7, -8], [0, 1])

def split_to_sets(inputs, labels, splits=(0.7, 0.15, 0.15)):
    ''' Takes inputs and labels and return three sets: train, test, and validate.
    Each set is a (name, inputs, labels) named tuple. '''

    sizes = [int(len(inputs)*split) for split in splits]

    curr, inp_splits, lab_splits = 0, [], []
    for size in sizes:
        inp_splits.append(inputs[curr:curr+size])
        lab_splits.append(labels[curr:curr+size])
        curr += size

    return DataSet("Train", inp_splits[0], lab_splits[0]), \
        DataSet("Test", inp_splits[1], lab_splits[1]), \
        DataSet("Validate", inp_splits[2], lab_splits[2])

####################################################################################################

class DataSet:
    def __init__(self, name, inputs, labels):
        self.name = name
        self.inputs = inputs
        self.labels = labels

####################################################################################################

def batch_iterator(ds, n):
    i = 0
    for i in range(0, len(ds.inputs)-n, n):
        yield ds.inputs[i:i+n], ds.labels[i:i+n]
    yield ds.inputs[i:], ds.labels[i:] # Last batch can be larger than n

####################################################################################################

def get_log_scaled_range(vmin, vmax, n_points, *, random):
    if random:
        steps = np.random.rand(n_points)
    else:
        steps = np.array(range(n_points)) / (n_points-1)
    lmin = log10(vmin)
    exps = lmin + (log10(vmax)-lmin)*steps
    return pow(10, exps)

####################################################################################################
