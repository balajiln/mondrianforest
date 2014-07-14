#!/usr/bin/env python
# contains a bunch of functions that are shared across multiple files
# ./mondrianforest_utils.py will give a detailed log


import sys
import math
import optparse
import cPickle as pickle
import numpy as np
import random
from itertools import izip
try:
    from scipy.special import gammaln, digamma
    from scipy.special import gdtrc         # required only for regression
    from scipy.optimize import fsolve       # required only for regression
    import scipy.stats
except:
    print 'Error loading scipy modules; might cause error later'
from copy import copy
try:
    import matplotlib.pyplot as plt       # uncomment if you need to plot
    from mpl_toolkits.mplot3d import Axes3D
except:
    print 'Error loading matplotlib; might cause error later'
from utils import hist_count, logsumexp, softmax, sample_multinomial, \
        sample_multinomial_scores, empty, assert_no_nan, linear_regression, \
        check_if_zero, check_if_one


def parser_add_common_options():
    parser = optparse.OptionParser()
    parser.add_option('--dataset', dest='dataset', default='toy',
            help='name of the dataset  [default: %default]')
    parser.add_option('--normalize_features', dest='normalize_features', default=0, type='int',
            help='do you want to normalize features in the range 0-1? (0=False, 1=True) [default: %default]')
    parser.add_option('--optype', dest='optype', default='class',
            help='nature of outputs in your dataset (class/real) '\
            'for (classification/regression)  [default: %default]')
    parser.add_option('--data_path', dest='data_path', default='../../process_data/',
            help='path of the dataset [default: %default]')
    parser.add_option('--debug', dest='debug', default='0', type='int',
            help='debug or not? (0=False, 1=everything, 2=special stuff only) [default: %default]')
    parser.add_option('--op_dir', dest='op_dir', default='results', 
            help='output directory for pickle files (NOTE: make sure directory exists) [default: %default]')
    parser.add_option('--tag', dest='tag', default='', 
            help='additional tag to identify results from a particular run [default: %default]' \
                    'tag=donottest reduces test time drastically (useful for profiling training time)')
    parser.add_option('--save', dest='save', default=0, type='int',
            help='do you wish to save the results? (1=True, 0=False) [default: %default]') 
    parser.add_option('-v', '--verbose',dest='verbose', default=1, type='int',
            help='verbosity level (0 is minimum, 4 is maximum) [default: %default]')
    parser.add_option('--init_id', dest='init_id', default=1, type='int',
            help='init_id (changes random seed for multiple initializations) [default: %default]')
    return parser


def parser_add_mf_options(parser):
    group = optparse.OptionGroup(parser, "Mondrian forest options")
    group.add_option('--n_mondrians', dest='n_mondrians', default=10, type='int',
            help='number of trees in mondrian forest [default: %default]')   
    group.add_option('--budget', dest='budget', default=-1, type='float',
            help='budget for mondrian tree prior (NOTE: budget=-1 will be treated as infinity) [default: %default]')   
    group.add_option('--discount_factor', dest='discount_factor', default=10, type='float',
            help='value of discount_factor parameter [default: %default] '
            'NOTE: actual discount parameter = discount_factor * num_dimensions')   
    group.add_option('--n_minibatches', dest='n_minibatches', default=1, type='int',
            help='number of minibatches [default: %default]')   
    group.add_option('--draw_mondrian', dest='draw_mondrian', default=0, type='int',
            help='do you want to draw mondrians? (0=False, 1=True) [default: %default] ')
    group.add_option('--store_every', dest='store_every', default=0, type='int',
            help='do you want to store mondrians at every iteration? (0=False, 1=True)')
    group.add_option('--bagging', dest='bagging', default=0, type='int',
            help='do you want to use bagging? (0=False, 1=True) [default: %default] ')
    parser.add_option_group(group)
    return parser


def parser_check_common_options(parser, settings):
    fail(parser, not(settings.save==0 or settings.save==1), 'save needs to be 0/1')
    fail(parser, not(settings.normalize_features==0 or settings.normalize_features==1), 'normalize_features needs to be 0/1')
    fail(parser, not(settings.optype=='real' or settings.optype=='class'), 'optype needs to be real/class')


def parser_check_mf_options(parser, settings):
    fail(parser, settings.n_mondrians < 1, 'number of mondrians needs to be >= 1')
    fail(parser, settings.discount_factor <= 0, 'discount_factor needs to be > 0')
    fail(parser, not(settings.budget == -1 or settings.budget > 0), 'budget needs to be > 0 or -1 (treated as INF)')
    fail(parser, settings.n_minibatches < 1, 'number of minibatches needs to be >= 1')
    fail(parser, not(settings.draw_mondrian==0 or settings.draw_mondrian==1), 'draw_mondrian needs to be 0/1')
    fail(parser, not(settings.store_every==0 or settings.store_every==1), 'store_every needs to be 0/1')
    fail(parser, not(settings.bagging==0 or settings.bagging==1), 'bagging needs to be 0/1')
    # added additional checks for MF
    fail(parser, not(settings.normalize_features==1), \
            'normalize_features needs to be 1 for mondrian forests')
    fail(parser, not(settings.optype=='class'), 'optype needs to be class (regression is not supported yet)')


def fail(parser, condition, msg):
    if condition:
        print msg
        print
        parser.print_help()
        sys.exit(1)


def reset_random_seed(settings):
    # Resetting random seed
    np.random.seed(settings.init_id * 1000)
    random.seed(settings.init_id * 1000)


def check_dataset(settings):
    classification_datasets = set(['satimage', 'usps', 'dna', 'dna-61-120', 'letter'])
    special_cases = settings.dataset[:3] == 'toy' or settings.dataset[:9] == 'halfmoons'
    if not special_cases:
        try:
            if settings.optype == 'class':
                assert(settings.dataset in classification_datasets)
        except AssertionError:
            print 'Invalid dataset for optype; dataset = %s, optype = %s' % \
                    (settings.dataset, settings.optype)
            raise AssertionError
    return special_cases


def load_data(settings):
    data = {}
    special_cases = check_dataset(settings)
    if not special_cases:
        data = pickle.load(open(settings.data_path + settings.dataset + '/' + \
                settings.dataset + '.p', "rb"))
    elif settings.dataset == 'toy-mf':
        data = load_toy_mf_data()
    elif settings.dataset[:9] == 'halfmoons':
        data = load_halfmoons(settings.dataset)
    else:
        print 'Unknown dataset: ' + settings.dataset
        raise Exception
    assert(not data['is_sparse'])
    try:
        if settings.normalize_features == 1:
            min_d = np.minimum(np.min(data['x_train'], 0), np.min(data['x_test'], 0))
            max_d = np.maximum(np.max(data['x_train'], 0), np.max(data['x_test'], 0))
            data['x_train'] -= min_d + 0.
            data['x_train'] /= (max_d - min_d)
            data['x_test'] -= min_d + 0.
            data['x_test'] /= (max_d - min_d)
    except AttributeError:
        # backward compatibility with code without normalize_features argument
        pass
    # ------ beginning of hack ----------
    is_mondrianforest = True
    n_minibatches = settings.n_minibatches
    if is_mondrianforest:
        # creates data['train_ids_partition']['current'] and data['train_ids_partition']['cumulative'] 
        #    where current[idx] contains train_ids in minibatch "idx", cumulative contains train_ids in all
        #    minibatches from 0 till idx  ... can be used in gen_train_ids_mf or here (see below for idx > -1)
        data['train_ids_partition'] = {'current': {}, 'cumulative': {}}
        train_ids = np.arange(data['n_train'])
        try:
            draw_mondrian = settings.draw_mondrian
        except AttributeError:
            draw_mondrian = False
        if is_mondrianforest and (not draw_mondrian):
            reset_random_seed(settings)
            np.random.shuffle(train_ids)
            # NOTE: shuffle should be the first call after resetting random seed
            #       all experiments would NOT use the same dataset otherwise
        train_ids_cumulative = np.arange(0)
        n_points_per_minibatch = data['n_train'] / n_minibatches
        assert n_points_per_minibatch > 0
        idx_base = np.arange(n_points_per_minibatch)
        for idx_minibatch in range(n_minibatches):
            is_last_minibatch = (idx_minibatch == n_minibatches - 1)
            idx_tmp = idx_base + idx_minibatch * n_points_per_minibatch
            if is_last_minibatch:
                # including the last (data[n_train'] % settings.n_minibatches) indices along with indices in idx_tmp
                idx_tmp = np.arange(idx_minibatch * n_points_per_minibatch, data['n_train'])
            train_ids_current = train_ids[idx_tmp]
            # print idx_minibatch, train_ids_current
            data['train_ids_partition']['current'][idx_minibatch] = train_ids_current
            train_ids_cumulative = np.append(train_ids_cumulative, train_ids_current)
            data['train_ids_partition']['cumulative'][idx_minibatch] = train_ids_cumulative
    return data


def add_stuff_2_settings(settings):
    settings.perf_dataset_keys = ['train', 'test']
    if settings.optype == 'class':
        settings.perf_store_keys = ['pred_prob']
        settings.perf_metrics_keys = ['log_prob', 'acc']
    else:
        settings.perf_store_keys = ['pred_mean', 'pred_prob']
        settings.perf_metrics_keys = ['log_prob', 'mse']


def get_name_metric(settings):
    name_metric = settings.perf_metrics_keys[1]
    assert(name_metric == 'mse' or name_metric == 'acc')
    return name_metric


def load_toy_data():
    n_dim = 2
    n_train_pc = 4
    n_class = 2
    n_train = n_train_pc * n_class
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    x_train = np.random.randn(n_train, n_dim)
    x_test = np.random.randn(n_train, n_dim)
    mag = 5
    for i, y_ in enumerate(y_train):
        if y_ == 0:
            x_train[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_train[i, :] += np.array([tmp, -tmp]) * mag
    for i, y_ in enumerate(y_test):
        if y_ == 0:
            x_test[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_test[i, :] += np.array([tmp, -tmp]) * mag
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    print data
    return data
 

def load_toy_mf_data():
    n_dim = 2
    n_class = 3
#    x_train = np.array([-1, -1, -2, -2, 1,1, 2,2, 3,3, 4,4, -1, 1, -2, 2, 2,-2, 3, -3])
#    x_train.shape = (10, 2)
#    x_test = np.array([0,0.5, -1,0.5, 10,10]) + 0.
#    x_test.shape=(3,2)
    #x_train = np.array([-1,-1, -2,-2, 1,1, 2,2, -1,1, -2, 2]) + 0.
    x_train = np.array([-0.5,-1, -2,-2, 1,0.5, 2,2, -1,1, -1.5, 1.5]) + 0.
    y_train = np.array([0, 0, 1, 1, 2, 2], dtype='int')
    #y_train *= 0
    x_train.shape = (6, 2)
    if False:
        plt.figure()
        plt.hold(True)
        plt.scatter(x_train[:2, 0], x_train[:2, 1], color='b')
        plt.scatter(x_train[2:4, 0], x_train[2:4, 1], color='r')
        plt.scatter(x_train[4:, 0], x_train[4:, 1], color='k')
        plt.savefig('toy-mf_dataset.pdf', type='pdf') 
        #plt.show()
    #x_test = np.array([0,0.5, -1,0.5]) + 0.
#    x_test = np.array([-0.5,-1.5, -0.5,-1.5, 0.5,0.5, 1.5,1.5, -0.5,0.5, -1.5,1.5]) + 0.
#    x_test.shape=(6,2)
    x_test = x_train
    #x_test = x_train * 1.25
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    y_test = np.array([0, 0, 1, 1, 2, 2], dtype='int')
#    print x_train, x_test, y_train, y_test
#    mag = 5
#    for i, y_ in enumerate(y_train):
#        if y_ == 0:
#            x_train[i, :] += np.sign(np.random.rand() - 0.5) * mag
#        else:
#            tmp = np.sign(np.random.rand() - 0.5) 
#            if n_dim == 2:
#                x_train[i, :] += np.array([tmp, -tmp]) * mag
#            else:
#                x_train[i, :] += np.array([tmp]) * mag
#    for i, x_ in enumerate(x_test):
#        sign = np.sign(np.prod(x_))
#        if sign > 0:
#            y_test[i] = 0
#        else:
#            y_test[i] = 1
#        if np.any(np.abs(x_) > 7):
#            y_test[i] = int(round(np.random.rand()))
#    print 'checking x_test = %s' % x_test
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
#    print 'training data = %s' % np.column_stack((x_train, y_train))
#    print 'testing data = %s' % np.column_stack((x_test, y_test))
    return data


def load_halfmoons(dataset):
    n_dim = 2
    n_class = 2
    if dataset == 'halfmoons':
        x_train = np.array([-3,0, -2,1, -1,2, 0,3, 1,2, 2,1, 3,0, -1.5,1.5, -0.5,0.5, 0.5,-0.5, 1.5,-1.5, 2.5,-0.5, 3.5,0.5, 4.5,1.5]) + 0.
        y_train = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype='int')
        x_train.shape = (14, 2)
        x_train[7:, 0] += 1.5
        x_train[7:, 1] -= 0.5
        i1 = np.arange(7)
        i2 = np.arange(7, 14)
    elif dataset == 'halfmoons2':
        n = 500
        x_train = np.random.rand(n, 2)
        #x_train = np.zeros((n, 2))
        #x_train[:, 0] = np.linspace(0, 1, n)
        i1 = np.arange(n / 2)
        i2 = np.arange(n / 2, n)
        x_train[i1, 0] = 2 * x_train[i1, 0] - 1
        x_train[i2, 0] = 2 * x_train[i2, 0]
        x_train[i1, 1] = 1 - x_train[i1, 0] * x_train[i1, 0]
        x_train[i2, 1] = (x_train[i2, 0] - 1) * (x_train[i2, 0] - 1) - 0.5
        x_train[:, 1] += 0.1 * np.random.randn(n)
        y_train = np.zeros(n, dtype='int')
        y_train[i2] = 1
    else:
        raise Exception
    if False:
        plt.figure()
        plt.hold(True)
        plt.scatter(x_train[i1, 0], x_train[i1, 1], color='b')
        plt.scatter(x_train[i2, 0], x_train[i2, 1], color='r')
        name = '%s_dataset.pdf' % dataset
        plt.savefig(name, type='pdf') 
        #plt.show()
    #x_test = np.array([0,0.5, -1,0.5]) + 0.
    #x_test = np.array([-0.5,-1.5, -0.5,-1.5]) + 0.
    #x_test.shape=(2,2)
    #y_test = np.array([0, 0], dtype='int')
    #x_test, y_test = x_train + 0.5, y_train
    x_test, y_test = x_train.copy(), y_train.copy()
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
#    print x_train, x_test, y_train, y_test
#    print 'checking x_test = %s' % x_test
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data

 
def get_tree_limits(p, data):
    d_extent = {}
    x = data['x_train']
    x1_min = np.min(x[:, 1])
    x1_max = np.max(x[:, 1])
    x0_min = np.min(x[:, 0])
    x0_max = np.max(x[:, 0])
    d_extent[0] = (x0_min, x0_max, x1_min, x1_max)
    if not p.node_info:
        return ([], [])
    non_leaf_max = max(p.node_info.keys())
    p_d = p.node_info
    hlines_list = []
    vlines_list = []
    for node_id in range(non_leaf_max + 1):
        if node_id not in p.node_info:
            continue
        x0_min, x0_max, x1_min, x1_max = d_extent[node_id]
        print p_d[node_id]
        feat_id, split, idx_split_global = p_d[node_id]
        if feat_id == 0:
            vlines_list.append([split, x1_min, x1_max])
            left_extent = (x0_min, split, x1_min, x1_max)
            right_extent = (split, x0_max, x1_min, x1_max)
        else:
            hlines_list.append([split, x0_min, x0_max])
            left_extent = (x0_min, x0_max, x1_min, split)
            right_extent = (x0_min, x0_max, split, x1_max)
        left, right = get_children_id(node_id)
        d_extent[left] = left_extent
        d_extent[right] = right_extent
    return (hlines_list, vlines_list)
            

def bootstrap(train_ids, settings=None):
    """ online bagging: each point is included Poisson(1) times """
    n = len(train_ids)
    cnt_all = np.random.poisson(1, n)
    op = []
    for train_id, cnt in izip(train_ids, cnt_all):
        op.extend([train_id] * cnt)
    return np.array(op)


class Param(object):
    def __init__(self, settings):
        self.budget = settings.budget


def get_filename_mf(settings):
    if settings.optype == 'class':
        param_str = '%s' % settings.alpha
    split_str = 'mf-budg-%s_nmon-%s_mini-%s_discount-%s' % (settings.budget, settings.n_mondrians, \
                                        settings.n_minibatches, settings.discount_factor)
    filename = settings.op_dir + '/' + '%s-%s-param-%s-init_id-%s-bag-%s-tag-%s.p' % \
            (settings.dataset, split_str, param_str, settings.init_id, \
                settings.bagging, settings.tag)
    return filename


def create_prediction_tree(tree, param, data, settings, all_nodes=False):
    init_prediction_tree(tree, settings)
    for node_id in tree.leaf_nodes:
        update_predictive_posterior_node(tree, param, data, settings, node_id)
    if all_nodes:
        for node_id in tree.non_leaf_nodes:
            update_predictive_posterior_node(tree, param, data, settings, node_id)


def init_prediction_tree(tree, settings):
    if settings.optype == 'class':
        tree.pred_prob = {}


def update_predictive_posterior_node(tree, param, data, settings, node_id):
    if settings.optype == 'class':
        #tmp = tree.counts[node_id] + float(param.alpha) / data['n_class']
        #tmp = tree.counts[node_id] + param.alpha_per_class
        tmp = tree.counts[node_id] + param.alpha_vec
        tree.pred_prob[node_id] = tmp / float(tmp.sum())


def compute_metrics_classification(y_test, pred_prob, do_not_compute_log_prob=False):
    acc, log_prob = 0.0, 0.0
    for n, y in enumerate(y_test):
        tmp = pred_prob[n, :]
        #pred = np.argmax(tmp)
        pred = random.choice(np.argwhere(tmp == np.amax(tmp)).flatten())    # randomly break ties
        acc += (pred == y)
        if not do_not_compute_log_prob:
            log_tmp_pred = math.log(tmp[y]) 
            try:
                assert(not np.isinf(abs(log_tmp_pred)))
            except AssertionError:
                'print abs(log_tmp_pred) = inf in compute_metrics_classification; tmp = '
                print tmp
                raise AssertionError
            log_prob += log_tmp_pred
    acc /= (n + 1)
    if not do_not_compute_log_prob:
        log_prob /= (n + 1)
    else:
        log_prob = -np.inf
    metrics = {'acc': acc, 'log_prob': log_prob}
    return metrics


def test_compute_metrics_classification():
    n = 100
    n_class = 10
    pred_prob = np.random.rand(n, n_class)
    y = np.ones(n)
    metrics = compute_metrics_classification(y, pred_prob)
    print 'chk if same: %s, %s' % (metrics['log_prob'], np.mean(np.log(pred_prob[:, 1])))
    assert(np.abs(metrics['log_prob']  - np.mean(np.log(pred_prob[:, 1]))) < 1e-10)
    pred_prob[:, 1] = 1e5
    metrics = compute_metrics_classification(y, pred_prob)
    assert np.abs(metrics['acc'] - 1) < 1e-3
    print 'chk if same: %s, 1.0' % (metrics['acc'])


def is_split_valid(split_chosen, x_min, x_max):
    try:
        assert(split_chosen > x_min)
        assert(split_chosen < x_max)
    except AssertionError:
        print 'split_chosen <= x_min or >= x_max'
        raise AssertionError


def evaluate_performance_tree(p, param, data, settings, x_test, y_test):
    create_prediction_tree(p, param, data, settings)
    pred_all = evaluate_predictions_fast(p, x_test, y_test, data, param, settings)
    pred_prob = pred_all['pred_prob']
    if settings.optype == 'class':
        metrics = compute_metrics_classification(y_test, pred_prob)
    else:
        pred_mean = pred_all['pred_mean']
        metrics = compute_metrics_regression(y_test, pred_mean, pred_prob)
    return (metrics)


def compute_metrics(y_test, pred_prob):
    acc, log_prob = 0.0, 0.0
    for n, y in enumerate(y_test):
        tmp = pred_prob[n, :]
        pred = np.argmax(tmp)
        acc += (pred == y_test[n])
        #log_tmp_pred = np.log(tmp[pred]) 
        log_tmp_pred = np.log(tmp[y]) 
        try:
            assert(not np.isinf(abs(log_tmp_pred)))
        except AssertionError:
            'print abs(log_tmp_pred) = inf in compute_metrics; tmp = '
            print tmp
            raise AssertionError
        log_prob += log_tmp_pred
    acc /= (n + 1)
    log_prob /= (n + 1)
    return (acc, log_prob)


def stop_split(train_ids, settings, data, cache):
    if (len(train_ids) <= settings.min_size):
        op = True
    else:
        op = no_valid_split_exists(data, cache, train_ids, settings)
    return op


def compute_dirichlet_normalizer(cnt, alpha=0.0, prior_term=None):
    """ cnt is np.array, alpha is concentration of Dirichlet prior 
        => alpha/K is the mass for each component of a K-dimensional Dirichlet
    """
    try:
        assert(len(cnt.shape) == 1)
    except AssertionError:
        print 'cnt should be a 1-dimensional np array'
        raise AssertionError
    n_class = float(len(cnt))
    if prior_term is None:
        #print 'recomputing prior_term'
        prior_term = gammaln(alpha) - n_class * gammaln(alpha / n_class)
    op = np.sum(gammaln(cnt + alpha / n_class)) - gammaln(np.sum(cnt) + alpha) \
            + prior_term
    return op


def compute_dirichlet_normalizer_fast(cnt, cache):
    """ cnt is np.array, alpha is concentration of Dirichlet prior 
        => alpha/K is the mass for each component of a K-dimensional Dirichlet
    """
    op = compute_gammaln_1(cnt, cache) - compute_gammaln_2(cnt.sum(), cache) \
            + cache['alpha_prior_term']
    return op


def evaluate_predictions(p, x, y, data, param):
    (pred, pred_prob) = p.predict(x, data['n_class'], param.alpha)
    (acc, log_prob) = compute_metrics(y, pred_prob)
    return (pred, pred_prob, acc, log_prob)


def init_left_right_statistics():
    return(None, None, {}, -np.inf, -np.inf)


def compute_left_right_statistics(data, param, cache, train_ids, feat_id_chosen, \
        split_chosen, settings, do_not_compute_loglik=False):
    cond = data['x_train'][train_ids, feat_id_chosen] <= split_chosen
    train_ids_left = train_ids[cond]
    train_ids_right = train_ids[~cond]
    cache_tmp = {}
    if settings.optype == 'class':
        range_n_class = cache['range_n_class']
        cnt_left_chosen = np.bincount(data['y_train'][train_ids_left], minlength=data['n_class'])
        cnt_right_chosen = np.bincount(data['y_train'][train_ids_right], minlength=data['n_class'])
        if not do_not_compute_loglik:
            loglik_left = compute_dirichlet_normalizer_fast(cnt_left_chosen, cache)
            loglik_right = compute_dirichlet_normalizer_fast(cnt_right_chosen, cache)
        else:
            loglik_left = loglik_right = -np.inf
        cache_tmp['cnt_left_chosen'] = cnt_left_chosen
        cache_tmp['cnt_right_chosen'] = cnt_right_chosen
    if settings.verbose >= 2:
        print 'feat_id_chosen = %s, split_chosen = %s' % (feat_id_chosen, split_chosen)
        print 'y (left) = %s\ny (right) = %s' % (data['y_train'][train_ids_left], \
                                                    data['y_train'][train_ids_right])
        print 'loglik (left) = %.2f, loglik (right) = %.2f' % (loglik_left, loglik_right)
    return(train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right)


def compute_entropy(cnts, alpha=0.0):
    """ returns the entropy of a multinomial distribution with 
        mean parameter \propto (cnts + alpha/len(cnts))
        entropy unit = nats """
    prob = cnts * 1.0 + alpha / len(cnts)
    prob /= float(np.sum(prob))
    entropy = 0.0
    for k in range(len(cnts)):
        if abs(prob[k]) > 1e-12:
            entropy -= prob[k] * np.log(prob[k])
    return entropy


def precompute_minimal(data, settings):
    param = empty()
    cache = {}
    assert settings.optype == 'class'
    if settings.optype == 'class':
        param.alpha = settings.alpha
        param.alpha_per_class = float(param.alpha) / data['n_class']
        cache['y_train_counts'] = hist_count(data['y_train'], range(data['n_class']))
        cache['range_n_class'] = range(data['n_class'])
        param.base_measure = (np.ones(data['n_class']) + 0.) / data['n_class']
        param.alpha_vec = param.base_measure * param.alpha
    return (param, cache)


def init_update_posterior_node_incremental(tree, data, param, settings, cache, node_id, train_ids_new, \
         do_not_compute_loglik=False, init_node_id=None):
    # init with sufficient statistics of init_node_id and add contributions of train_ids_new
    if settings.optype == 'class':
        if init_node_id is None:
            tree.counts[node_id] = 0
        else:
            tree.counts[node_id] = tree.counts[init_node_id] + 0
    else:
        if init_node_id is None:
            tree.sum_y[node_id] = 0
            tree.sum_y2[node_id] = 0
            tree.n_points[node_id] = 0
        else:
            tree.sum_y[node_id] = tree.sum_y[init_node_id] + 0 
            tree.sum_y2[node_id] = tree.sum_y2[init_node_id] + 0
            tree.n_points[node_id] = tree.n_points[init_node_id] + 0
    update_posterior_node_incremental(tree, data, param, settings, cache, node_id, train_ids_new, do_not_compute_loglik)


def update_posterior_node_incremental(tree, data, param, settings, cache, node_id, train_ids_new, \
        do_not_compute_loglik=False):
    y_train_new = data['y_train'][train_ids_new]
    if settings.optype == 'class':
        #tree.counts[node_id] += hist_count(y_train_new, cache['range_n_class'])
        tree.counts[node_id] += np.bincount(y_train_new, minlength=data['n_class'])
        if not do_not_compute_loglik:
            tree.loglik[node_id] = compute_dirichlet_normalizer_fast(tree.counts[node_id], cache)
    else:
        #tree.sum_y[node_id], tree.sum_y2[node_id], tree.n_points[node_id] = get_reg_stats(y_train_current)
        sum_y_new, sum_y2_new, n_points_new = get_reg_stats(y_train_new)
        tree.sum_y[node_id] += sum_y_new
        tree.sum_y2[node_id] += sum_y2_new
        tree.n_points[node_id] += n_points_new
        tree.loglik[node_id], tree.param_n[node_id] = compute_normal_normalizer(tree.sum_y[node_id], \
                tree.sum_y2[node_id], tree.n_points[node_id], param, cache, settings)


def main():
    print 'Running test_compute_metrics_classification()'
    test_compute_metrics_classification()


if __name__ == "__main__":
    main()
