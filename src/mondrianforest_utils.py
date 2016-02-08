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
import scipy.io
from warnings import warn
try:
    from scipy.special import gammaln, digamma
    from scipy.special import gdtrc         # required only for regression
    from scipy.optimize import fsolve       # required only for regression
    import scipy.stats
    from scipy.stats.stats import pearsonr
except:
    print 'Error loading scipy modules; might cause error later'
from copy import copy
try:
    from sklearn import feature_selection
except:
    print 'Error loading sklearn; might cause error later'
try:
    import matplotlib.pyplot as plt       # uncomment if you need to plot
    from mpl_toolkits.mplot3d import Axes3D
except:
    print 'Error loading matplotlib; might cause error later'
from utils import hist_count, logsumexp, softmax, sample_multinomial, \
        sample_multinomial_scores, empty, assert_no_nan, linear_regression, \
        check_if_zero, check_if_one, logsumexp_array


class Forest(object):
    def __init__(self):
        pass

    def evaluate_predictions(self, data, x, y, settings, param, weights, print_results=True):
        if settings.optype == 'class':
            pred_forest = {'pred_prob': np.zeros((x.shape[0], data['n_class']))}
        else:
            pred_forest = {'pred_mean': np.zeros(x.shape[0]), 'pred_var': np.zeros(x.shape[0]), \
                            'pred_prob': np.zeros(x.shape[0]), 'log_pred_prob': -np.inf*np.ones(x.shape[0]), \
                            'pred_sample': np.zeros(x.shape[0])}
        if settings.debug:
            check_if_one(weights.sum())
        if settings.verbose >= 2:
            print 'weights = \n%s' % weights
        for i_t, tree in enumerate(self.forest):
            pred_all = evaluate_predictions_tree(tree, x, y, data, param, settings)
            if settings.optype == 'class':
                # doesn't make sense to average predictive probabilities across trees for real outputs
                pred_prob = pred_all['pred_prob']
                pred_forest['pred_prob'] += weights[i_t] * pred_prob
            elif settings.optype == 'real':
                # skipping pred_prob for real outputs
                pred_forest['pred_mean'] += weights[i_t] * pred_all['pred_mean']
                pred_forest['pred_var'] += weights[i_t] * pred_all['pred_second_moment']
                pred_forest['pred_sample'] += weights[i_t] * pred_all['pred_sample']
                pred_forest['log_pred_prob'] = logsumexp_array(pred_forest['log_pred_prob'], \
                        np.log(weights[i_t]) + pred_all['log_pred_prob'])
        if settings.optype == 'real':
            pred_forest['pred_var'] -= pred_forest['pred_mean'] ** 2
            pred_forest['pred_prob'] = np.exp(pred_forest['log_pred_prob'])
            # NOTE: logpdf takes in variance
            log_prob2 = compute_gaussian_logpdf(pred_forest['pred_mean'], pred_forest['pred_var'], y)
            if settings.verbose >= 1:
                print 'log_prob (using Gaussian approximation) = %f' % np.mean(log_prob2)
                print 'log_prob (using mixture of Gaussians) = %f' % np.mean(pred_forest['log_pred_prob'])
            try:
                assert np.all(pred_forest['pred_prob'] > 0.)
            except AssertionError:
                print 'pred prob not > 0'
                print 'min value = %s' % np.min(pred_forest['pred_prob'])
                print 'sorted array = %s' % np.sort(pred_forest['pred_prob'])
                # raise AssertionError
        if settings.debug and settings.optype == 'class':
            check_if_zero(np.mean(np.sum(pred_forest['pred_prob'], axis=1) - 1))
        if settings.optype == 'class':
            # True ignores log prob computation
            metrics = compute_metrics_classification(y, pred_forest['pred_prob'], True)
        else:
            metrics = compute_metrics_regression(y, pred_forest['pred_mean'], pred_forest['pred_prob'])
            if settings.optype == 'real':
                metrics['log_prob2'] = log_prob2
        if print_results:
            if settings.optype == 'class':
                print 'Averaging over all trees, accuracy = %f' % metrics['acc']
            else:
                print 'Averaging over all trees, mse = %f, rmse = %f, log_prob = %f' % (metrics['mse'], \
                        math.sqrt(metrics['mse']), metrics['log_prob'])
        return (pred_forest, metrics)


def evaluate_predictions_tree(tree, x, y, data, param, settings):
    if settings.optype == 'class':
        pred_prob = tree.predict_class(x, data['n_class'], param, settings)
        pred_all = {'pred_prob': pred_prob}
    else:
        pred_mean, pred_var, pred_second_moment, log_pred_prob, pred_sample = \
                tree.predict_real(x, y, param, settings)
        pred_all =  {'log_pred_prob': log_pred_prob, 'pred_mean': pred_mean, \
                        'pred_second_moment': pred_second_moment, 'pred_var': pred_var, \
                        'pred_sample': pred_sample}
    return pred_all


def compute_gaussian_pdf(e_x, e_x2, x):
    variance = np.maximum(0, e_x2 - e_x ** 2)
    sd = np.sqrt(variance)
    z = (x - e_x) / sd
    # pdf = np.exp(-(z**2) / 2.) / np.sqrt(2*math.pi) / sd
    log_pdf = -0.5*(z**2) -np.log(sd) -0.5*np.log(2*math.pi)
    pdf = np.exp(log_pdf)
    return pdf


def compute_gaussian_logpdf(e_x, variance, x):
    assert np.all(variance > 0)
    sd = np.sqrt(variance)
    z = (x - e_x) / sd
    log_pdf = -0.5*(z**2) -np.log(sd) -0.5*np.log(2*math.pi)
    return log_pdf


def parser_add_common_options():
    parser = optparse.OptionParser()
    parser.add_option('--dataset', dest='dataset', default='toy-mf',
            help='name of the dataset  [default: %default]')
    parser.add_option('--normalize_features', dest='normalize_features', default=1, type='int',
            help='do you want to normalize features in 0-1 range? (0=False, 1=True) [default: %default]')
    parser.add_option('--select_features', dest='select_features', default=0, type='int',
            help='do you wish to apply feature selection? (1=True, 0=False) [default: %default]') 
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
                    ' tag=donottest reduces test time drastically (useful for profiling training time)')
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
            help='budget for mondrian tree prior [default: %default]' \
                    ' NOTE: budget=-1 will be treated as infinity')   
    group.add_option('--discount_factor', dest='discount_factor', default=10, type='float',
            help='value of discount_factor parameter for HNSP (optype=class) [default: %default] '
            'NOTE: actual discount parameter = discount_factor * num_dimensions')   
    group.add_option('--n_minibatches', dest='n_minibatches', default=1, type='int',
            help='number of minibatches [default: %default]')   
    group.add_option('--draw_mondrian', dest='draw_mondrian', default=0, type='int',
            help='do you want to draw mondrians? (0=False, 1=True) [default: %default] ')
    group.add_option('--smooth_hierarchically', dest='smooth_hierarchically', default=1, type='int',
            help='do you want to smooth hierarchically? (0=False, 1=True)')
    group.add_option('--store_every', dest='store_every', default=0, type='int',
            help='do you want to store mondrians at every iteration? (0=False, 1=True)')
    group.add_option('--bagging', dest='bagging', default=0, type='int',
            help='do you want to use bagging? (0=False) [default: %default] ')
    group.add_option('--min_samples_split', dest='min_samples_split', default=2, type='int',
            help='the minimum number of samples required to split an internal node ' \
                    '(used only for optype=real) [default: %default]')
    parser.add_option_group(group)
    return parser


def parser_check_common_options(parser, settings):
    fail(parser, not(settings.save==0 or settings.save==1), 'save needs to be 0/1')
    fail(parser, not(settings.smooth_hierarchically==0 or settings.smooth_hierarchically==1), \
            'smooth_hierarchically needs to be 0/1')
    fail(parser, not(settings.normalize_features==0 or settings.normalize_features==1), 'normalize_features needs to be 0/1')
    fail(parser, not(settings.optype=='real' or settings.optype=='class'), 'optype needs to be real/class')


def parser_check_mf_options(parser, settings):
    fail(parser, settings.n_mondrians < 1, 'number of mondrians needs to be >= 1')
    fail(parser, settings.discount_factor <= 0, 'discount_factor needs to be > 0')
    fail(parser, not(settings.budget == -1 or settings.budget > 0), 'budget needs to be > 0 or -1 (treated as INF)')
    fail(parser, settings.n_minibatches < 1, 'number of minibatches needs to be >= 1')
    fail(parser, not(settings.draw_mondrian==0 or settings.draw_mondrian==1), 'draw_mondrian needs to be 0/1')
    fail(parser, not(settings.store_every==0 or settings.store_every==1), 'store_every needs to be 0/1')
    fail(parser, not(settings.bagging==0), 'bagging=1 not supported; please set bagging=0')
    fail(parser, settings.min_samples_split < 1, 'min_samples_split needs to be > 1')
    # added additional checks for MF
    if settings.normalize_features != 1:
        warn('normalize_features not equal to 1; mondrian forests assume that features are on the same scale')


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
    regression_datasets = set(['housing', 'kin40k'])
    special_cases = settings.dataset[:3] == 'toy' or settings.dataset[:4] == 'rsyn' \
            or settings.dataset[:8] == 'ctslices' or settings.dataset[:3] == 'msd' \
            or settings.dataset[:6] == 'houses' or settings.dataset[:9] == 'halfmoons' \
            or settings.dataset[:3] == 'sim' or settings.dataset == 'navada' \
            or settings.dataset[:3] == 'msg' or settings.dataset[:14] == 'airline-delays' \
            or settings.dataset == 'branin'
    if not special_cases:
        try:
            if settings.optype == 'class':
                assert(settings.dataset in classification_datasets)
            else:
                assert(settings.dataset in regression_datasets)
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
    elif settings.dataset == 'msg-4dim':
        data = load_msg_data()
    elif settings.dataset[:9] == 'halfmoons':
        data = load_halfmoons(settings.dataset)
    elif settings.dataset[:4] == 'rsyn' or settings.dataset[:8] == 'ctslices' \
            or settings.dataset[:6] == 'houses' or settings.dataset[:3] == 'msd':
        data = load_rgf_datasets(settings)
    elif settings.dataset[:13] == 'toy-hypercube':
        n_dim = int(settings.dataset[14:])
        data = load_toy_hypercube(n_dim, settings, settings.optype == 'class')
    elif settings.dataset[:14] == 'airline-delays':
        filename = settings.data_path + 'airline-delays/' + settings.dataset + '.p'
        data = pickle.load(open(filename, 'rb'))
    else:
        print 'Unknown dataset: ' + settings.dataset
        raise Exception
    assert(not data['is_sparse'])
    try:
        if settings.normalize_features == 1:
            min_d = np.minimum(np.min(data['x_train'], 0), np.min(data['x_test'], 0))
            max_d = np.maximum(np.max(data['x_train'], 0), np.max(data['x_test'], 0))
            range_d = max_d - min_d
            idx_range_d_small = range_d <= 0.   # find columns where all features are identical
            if data['n_dim'] > 1:
                range_d[idx_range_d_small] = 1e-3   # non-zero value just to prevent division by 0
            elif idx_range_d_small:
                range_d = 1e-3
            data['x_train'] -= min_d + 0.
            data['x_train'] /= range_d
            data['x_test'] -= min_d + 0.
            data['x_test'] /= range_d
    except AttributeError:
        # backward compatibility with code without normalize_features argument
        pass
    if settings.select_features:
        if settings.optype == 'real':
            scores, _ = feature_selection.f_regression(data['x_train'], data['y_train'])
        scores[np.isnan(scores)] = 0.   # FIXME: setting nan scores to 0. Better alternative?
        scores_sorted, idx_sorted = np.sort(scores), np.argsort(scores)
        flag_relevant = scores_sorted > (scores_sorted[-1] * 0.05)  # FIXME: better way to set threshold? 
        idx_feat_selected = idx_sorted[flag_relevant]
        assert len(idx_feat_selected) >= 1
        print scores
        print scores_sorted
        print idx_sorted
        # plt.plot(scores_sorted)
        # plt.show()
        if False:
            data['x_train'] = data['x_train'][:, idx_feat_selected]
            data['x_test'] = data['x_test'][:, idx_feat_selected]
        else:
            data['x_train'] = np.dot(data['x_train'], np.diag(scores)) 
            data['x_test'] = np.dot(data['x_test'], np.diag(scores))
        data['n_dim'] = data['x_train'].shape[1]
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


def get_correlation(X, y):
    scores = np.zeros(X.shape[1])
    for i_col in np.arange(X.shape[1]):
        x = X[:, i_col]
        scores[i_col] = np.abs(pearsonr(x, y)[0])
    return scores


def load_toy_hypercube(n_dim, settings, class_output=False):
    n_train = n_test = 10 * (2 ** n_dim)
    reset_random_seed(settings)
    x_train, y_train, f_train, f_values = gen_hypercube_data(n_train, n_dim, class_output)
    x_test, y_test, f_test, f_values = gen_hypercube_data(n_test, n_dim, class_output, f_values)
    data = {'x_train': x_train, 'y_train': y_train, \
            'f_train': f_train, 'f_test': f_test, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    if class_output:
        data['n_class'] = 2 ** n_dim
    return data


def gen_hypercube_data(n_points, n_dim, class_output, f_values=None):
    # synthetic hypercube-like dataset 
    # x-values of data points are close to vertices of a hypercube
    # y-value of data point is different
    y_sd = 0.
    x_sd = 0.1 
    mag = 3
    x = x_sd * np.random.randn(n_points, n_dim)
    n_vertices = 2 ** n_dim
    #i = np.random.randint(0, n_vertices, n_points)
    i = np.arange(n_vertices).repeat(n_points / n_vertices)     # equal distribution
    offsets = np.zeros((n_vertices, n_dim))
    for d in range(n_dim):
        tmp = np.ones(2**(n_dim-d))
        tmp[:2**(n_dim-d-1)] = -1
        offsets[:, d] = np.tile(tmp, (1, 2**d))[0]
    x += offsets[i, :]
    y = np.zeros(n_points)
    #f = np.zeros(n_points)
    if class_output:
        y = f = i
    else:
        if f_values is None:
            # generate only for training data
            f_values = np.random.randn(n_vertices) * mag
        f = f_values[i]
        y = f + y_sd * np.random.randn(n_points)
    return (x, y, f, f_values)


def load_msg_data():
    mat = scipy.io.loadmat('wittawat/demo_uncertainty_msgs_4d.mat')
    x_test = np.vstack((mat['Xte1'], mat['Xte2']))
    n_test1 = mat['Xte1'].shape[0]
    y_test = np.nan * np.ones(x_test.shape[0])
    data = {'x_train': mat['Xtr'], 'y_train': np.ravel(mat['Ytr']), \
            'x_test': x_test, 'y_test': y_test, \
            'x_test1': mat['Xte1'], 'x_test2': mat['Xte2'], \
            'n_test1': n_test1, \
            'y_test1': y_test, 'y_test2': y_test, \
            'n_train': mat['Xtr'].shape[0], 'n_test': x_test.shape[0], \
            'n_dim': x_test.shape[1], 'is_sparse': False}
    return data


def add_stuff_2_settings(settings):
    settings.perf_dataset_keys = ['train', 'test']
    if settings.optype == 'class':
        settings.perf_store_keys = ['pred_prob']
        settings.perf_metrics_keys = ['log_prob', 'acc']
    else:
        settings.perf_store_keys = ['pred_mean', 'pred_prob']
        settings.perf_metrics_keys = ['log_prob', 'mse']
    settings.name_metric = get_name_metric(settings)


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
    x_train = np.array([-0.5,-1, -2,-2, 1,0.5, 2,2, -1,1, -1.5, 1.5]) + 0.
    y_train = np.array([0, 0, 1, 1, 2, 2], dtype='int')
    x_train.shape = (6, 2)
    if False:
        plt.figure()
        plt.hold(True)
        plt.scatter(x_train[:2, 0], x_train[:2, 1], color='b')
        plt.scatter(x_train[2:4, 0], x_train[2:4, 1], color='r')
        plt.scatter(x_train[4:, 0], x_train[4:, 1], color='k')
        plt.savefig('toy-mf_dataset.pdf', type='pdf') 
    x_test = x_train
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    y_test = np.array([0, 0, 1, 1, 2, 2], dtype='int')
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
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
    x_test, y_test = x_train.copy(), y_train.copy()
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data

 
def load_rgf_datasets(settings):
    filename_train = settings.data_path + 'exp-data' + '/' + settings.dataset
    filename_test = filename_train[:-3]
    x_train = np.loadtxt(filename_train + '.train.x')
    y_train = np.loadtxt(filename_train + '.train.y')
    x_test = np.loadtxt(filename_test + '.test.x')
    y_test = np.loadtxt(filename_test + '.test.y')
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    n_dim = x_train.shape[1]
    data = {'x_train': x_train, 'y_train': y_train, \
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
    else:
        param_str = ''
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
        tmp = tree.counts[node_id] + param.alpha_vec
        tree.pred_prob[node_id] = tmp / float(tmp.sum())
    else:
        tree.pred_mean[node_id] = tree.sum_y[node_id] / float(tree.n_points[node_id])


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


def compute_metrics_regression(y_test, pred_mean, pred_prob=None):
    # print 'y_test: ', y_test[:5]
    # print 'pred_mean: ', pred_mean[:5]
    mse = np.mean((y_test - pred_mean) ** 2)
    log_prob = np.mean(np.log(pred_prob))
    metrics = {'mse': mse, 'log_prob': log_prob}
    return metrics


def test_compute_metrics_regression():
    n = 100
    pred_prob = np.random.rand(n)
    y = np.random.randn(n)
    pred = np.ones(n)
    metrics = compute_metrics_regression(y, pred, pred_prob)
    print 'chk if same: %s, %s' % (metrics['mse'], np.mean((y - 1) ** 2))
    assert np.abs(metrics['mse'] - np.mean((y - 1) ** 2)) < 1e-3


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
        split_chosen, settings):
    cond = data['x_train'][train_ids, feat_id_chosen] <= split_chosen
    train_ids_left = train_ids[cond]
    train_ids_right = train_ids[~cond]
    cache_tmp = {}
    if settings.optype == 'class':
        range_n_class = cache['range_n_class']
        cnt_left_chosen = np.bincount(data['y_train'][train_ids_left], minlength=data['n_class'])
        cnt_right_chosen = np.bincount(data['y_train'][train_ids_right], minlength=data['n_class'])
        cache_tmp['cnt_left_chosen'] = cnt_left_chosen
        cache_tmp['cnt_right_chosen'] = cnt_right_chosen
    else:
        cache_tmp['sum_y_left'] = np.sum(data['y_train'][train_ids_left])
        cache_tmp['sum_y2_left'] = np.sum(data['y_train'][train_ids_left] ** 2)
        cache_tmp['n_points_left'] = len(train_ids_left)
        cache_tmp['sum_y_right'] = np.sum(data['y_train'][train_ids_right])
        cache_tmp['sum_y2_right'] = np.sum(data['y_train'][train_ids_right] ** 2)
        cache_tmp['n_points_right'] = len(train_ids_right)
    if settings.verbose >= 2:
        print 'feat_id_chosen = %s, split_chosen = %s' % (feat_id_chosen, split_chosen)
        print 'y (left) = %s\ny (right) = %s' % (data['y_train'][train_ids_left], \
                                                    data['y_train'][train_ids_right])
    return(train_ids_left, train_ids_right, cache_tmp)


def get_reg_stats(y):
    # y is a list of numbers, get_reg_stats(y) returns stats required for computing regression likelihood
    y_ = np.array(y)
    sum_y = float(np.sum(y_))
    n_points = len(y_)
    sum_y2 = float(np.sum(pow(y_, 2)))
    return (sum_y, sum_y2, n_points)


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
    if settings.optype == 'class':
        param.alpha = settings.alpha
        param.alpha_per_class = float(param.alpha) / data['n_class']
        cache['y_train_counts'] = hist_count(data['y_train'], range(data['n_class']))
        cache['range_n_class'] = range(data['n_class'])
        param.base_measure = (np.ones(data['n_class']) + 0.) / data['n_class']
        param.alpha_vec = param.base_measure * param.alpha
    else:
        cache['sum_y'] = float(np.sum(data['y_train']))
        cache['sum_y2'] = float(np.sum(data['y_train'] ** 2))
        cache['n_points'] = len(data['y_train'])
        warn('initializing prior mean and precision to their true values')
        # FIXME: many of the following are relevant only for mondrian forests
        param.prior_mean = np.mean(data['y_train'])
        param.prior_variance = np.var(data['y_train'])
        param.prior_precision = 1.0 / param.prior_variance
        if not settings.smooth_hierarchically:
            param.noise_variance = 0.01     # FIXME: hacky
        else:
            K = min(1000, data['n_train'])     # FIXME: measurement noise set to fraction of unconditional variance
            param.noise_variance = param.prior_variance / (1. + K)  # assume noise variance = prior_variance / (2K)
            # NOTE: max_split_cost scales inversely with the number of dimensions
        param.variance_coef = 2.0 * param.prior_variance
        param.sigmoid_coef = data['n_dim']  / (2.0 * np.log2(data['n_train']))
        param.noise_precision = 1.0 / param.noise_variance
    return (param, cache)


def init_update_posterior_node_incremental(tree, data, param, settings, cache, node_id, train_ids_new, \
         init_node_id=None):
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
    update_posterior_node_incremental(tree, data, param, settings, cache, node_id, train_ids_new)


def update_posterior_node_incremental(tree, data, param, settings, cache, node_id, train_ids_new):
    y_train_new = data['y_train'][train_ids_new]
    if settings.optype == 'class':
        tree.counts[node_id] += np.bincount(y_train_new, minlength=data['n_class'])
    else:
        sum_y_new, sum_y2_new, n_points_new = get_reg_stats(y_train_new)
        tree.sum_y[node_id] += sum_y_new
        tree.sum_y2[node_id] += sum_y2_new
        tree.n_points[node_id] += n_points_new


def main():
    print 'Running test_compute_metrics_classification()'
    test_compute_metrics_classification()


if __name__ == "__main__":
    main()
