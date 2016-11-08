import gzip
import cPickle as pickle

def load(filename):
    with gzip.open(filename) as f:
        train_set, valid_set, test_set = pickle.load(f)

if __name__ == "__main__":
