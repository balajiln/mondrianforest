#!/usr/bin/env python

# script to extract dimensions 61-120 of the dna dataset
# According to https://www.sgi.com/tech/mlc/db/DNA.names,
# " Hint.   Much better performance is generally observed if attributes
#           closest to the junction are used. In the StatLog version, 
#           this means using attributes A61 to A120 only."

import cPickle as pickle
import sys

data = pickle.load(open('../dna/dna.p', 'rb'))
data['x_train'] = data['x_train'][:, 60:120]
data['x_test'] = data['x_test'][:, 60:120]
data['n_dim'] = 60

pickle.dump(data, open('dna-61-120.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
