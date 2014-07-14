#!/usr/bin/env python
# input dataset is pickle file (to retain same train/test split)
# example: ./convert_pickle_2_onlinerf_format.py dna-61-120
#
# Output is slightly different from LIBSVM format
# - need to add header of #Samples #Features #Classes #FeatureMinIndex to the files
# - class indices need to start from 0

import sys
import os
import cPickle as pickle
from itertools import izip

name = sys.argv[1]

data = pickle.load(open(name + '/' + name + '.p', 'rb'))

d_n_dim = {'magic04': 10, 'pendigits': 16, 'dna-61-120': 60}
d_n_class = {'magic04': 2, 'pendigits': 10, 'dna-61-120': 3}

feat_id_start = 1

def print_file(x, y, name, op_name):
    op = open(name + '/' + op_name, 'w')
    print>>op, '%s %s %s %s' % (len(y), d_n_dim[name], d_n_class[name], feat_id_start)
    for x_, y_ in izip(x, y):
        s = ' '.join(['%d:%f' % (i+1, x__) for i, x__ in enumerate(x_)])
        print>>op, '%s %s' % (y_, s)

print_file(data['x_train'], data['y_train'], name, name + '.orf.train') 
print_file(data['x_test'], data['y_test'], name, name + '.orf.test') 
