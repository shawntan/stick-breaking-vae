import gzip
import cPickle as pickle
import numpy as np


def load(filename):
    data_dir = "/data/lisa/data/mnist_binarized/"
    path_tr = data_dir + 'binarized_mnist_train.amat'
    path_va = data_dir + 'binarized_mnist_valid.amat'
    path_te = data_dir + 'binarized_mnist_test.amat'
    train_x = np.loadtxt(path_tr).astype('float32')
    valid_x = np.loadtxt(path_va).astype('float32')
    test_x = np.loadtxt(path_te).astype('float32')
    return ((train_x, None),
            (valid_x, None),
            (test_x, None))
#    with gzip.open(filename) as f:
#        train_set, valid_set, test_set = pickle.load(f)
#        return ((train_set[0] > 0.5, train_set[1]),
#                (valid_set[0] > 0.5, valid_set[1]),
#                (test_set[0] > 0.5, test_set[1]))

load("")
