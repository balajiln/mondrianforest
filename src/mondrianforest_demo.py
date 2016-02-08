#!/usr/bin/env python

import numpy as np
import pprint as pp     # pretty printing module
from matplotlib import pyplot as plt        # required only for plotting results
from mondrianforest_utils import load_data, reset_random_seed, precompute_minimal 
from mondrianforest import process_command_line, MondrianForest

PLOT = False

settings = process_command_line()
print 'Current settings:'
pp.pprint(vars(settings))

# Resetting random seed
reset_random_seed(settings)

# Loading data
data = load_data(settings)

param, cache = precompute_minimal(data, settings)

mf = MondrianForest(settings, data)

print '\nminibatch\tmetric_train\tmetric_test\tnum_leaves'

for idx_minibatch in range(settings.n_minibatches):
    train_ids_current_minibatch = data['train_ids_partition']['current'][idx_minibatch]
    if idx_minibatch == 0:
        # Batch training for first minibatch
        mf.fit(data, train_ids_current_minibatch, settings, param, cache)
    else:
        # Online update
        mf.partial_fit(data, train_ids_current_minibatch, settings, param, cache)

    # Evaluate
    weights_prediction = np.ones(settings.n_mondrians) * 1.0 / settings.n_mondrians
    train_ids_cumulative = data['train_ids_partition']['cumulative'][idx_minibatch]
    pred_forest_train, metrics_train = \
        mf.evaluate_predictions(data, data['x_train'][train_ids_cumulative, :], \
        data['y_train'][train_ids_cumulative], \
        settings, param, weights_prediction, False)
    pred_forest_test, metrics_test = \
        mf.evaluate_predictions(data, data['x_test'], data['y_test'], \
        settings, param, weights_prediction, False)
    name_metric = settings.name_metric     # acc or mse
    metric_train = metrics_train[name_metric]
    metric_test = metrics_test[name_metric]
    tree_numleaves = np.zeros(settings.n_mondrians)
    for i_t, tree in enumerate(mf.forest):
        tree_numleaves[i_t] = len(tree.leaf_nodes)
    forest_numleaves = np.mean(tree_numleaves)
    print '%9d\t%.3f\t\t%.3f\t\t%.3f' % (idx_minibatch, metric_train, metric_test, forest_numleaves)

print '\nFinal forest stats:'
tree_stats = np.zeros((settings.n_mondrians, 2))
tree_average_depth = np.zeros(settings.n_mondrians)
for i_t, tree in enumerate(mf.forest):
    tree_stats[i_t, -2:] = np.array([len(tree.leaf_nodes), len(tree.non_leaf_nodes)])
    tree_average_depth[i_t] = tree.get_average_depth(settings, data)[0]
print 'mean(num_leaves) = %.1f, mean(num_non_leaves) = %.1f, mean(tree_average_depth) = %.1f' \
        % (np.mean(tree_stats[:, -2]), np.mean(tree_stats[:, -1]), np.mean(tree_average_depth))
print 'n_train = %d, log_2(n_train) = %.1f, mean(tree_average_depth) = %.1f +- %.1f' \
        % (data['n_train'], np.log2(data['n_train']), np.mean(tree_average_depth), np.std(tree_average_depth))

if PLOT:
    plt.figure(1)
    plt.hold(True)
    plt.scatter(data['x_train'][:, 0], data['y_train'], color='r')
    x_test = data['x_test'][:, 0]
    pred_mean = pred_forest_test['pred_mean']
    pred_sd = pred_forest_test['pred_var'] ** 0.5
    plt.errorbar(x_test, pred_mean, pred_sd, fmt='b.')
    
    plt.show()
