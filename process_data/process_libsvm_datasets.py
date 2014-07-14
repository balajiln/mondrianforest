#!/usr/bin/env python
# usage: ./process_libsvm_datasets.py name filename_train filename_test filename_val
# name is the folder name as well as the pickled filename
# filename_train and filename_test are assumed to exist in the folder name
# filename_val is optional (validation dataset will be added to training data if supplied)

import sys
import numpy as np
import cPickle as pickle
from sklearn.datasets import load_svmlight_file

def get_x_y(filename, name):
    (x, y) = load_svmlight_file(filename)
    x = x.toarray()
    if name == 'pendigits-svm':
        x = x[:, 1:]
    y = y.astype('int')
    if name != 'pendigits-svm':
        y -= 1
    return (x, y)

name, filename_train, filename_test = sys.argv[1:4]

data = {}

x, y = get_x_y(name + '/' + filename_train, name)
data['x_train'] = x
data['y_train'] = y
if len(sys.argv) > 4:
    filename_val = sys.argv[4]
    x, y = get_x_y(name + '/' + filename_val, name)
    data['x_train'] = np.vstack((data['x_train'], x))
    data['y_train'] = np.append(data['y_train'], y)
data['n_train'] = data['x_train'].shape[0]
assert len(data['y_train']) == data['n_train']

x, y = get_x_y(name + '/' + filename_test, name)
data['x_test'] = x
data['n_test'] = x.shape[0]
data['y_test'] = y

data['n_dim'] = x.shape[1]
data['n_class'] = len(np.unique(y))
try:
    assert data['n_class'] == max(np.unique(y)) + 1
except AssertionError:
    print 'np.unique(y) = %s' % np.unique(y)
    raise AssertionError
data['is_sparse'] = False

print 'name = %10s, n_dim = %5d, n_class = %5d, n_train = %5d, n_test = %5d' \
        % (name, data['n_dim'], data['n_class'], data['n_train'], data['n_test'])

pickle.dump(data, open(name + '/' + name + ".p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
