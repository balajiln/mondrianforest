#!/usr/bin/env python

import numpy as np
import math


def compute_kl(mean_1, log_prec_1, mean_2, log_prec_2):
    """
    compute KL between two Gaussian distributions
    """
    mean_1 = np.asarray(mean_1)
    log_prec_1 = np.asarray(log_prec_1)
    mean_2 = np.asarray(mean_2)
    log_prec_2 = np.asarray(log_prec_2)
    len_1 = len(mean_1)
    len_2 = len(mean_2)
    len_max = max(len_1, len_2)
    var_1, var_2 = np.exp(-log_prec_1), np.exp(-log_prec_2)     # computationally more stable?
    kl = 0.5 * ( log_prec_1 - log_prec_2 - 1 + np.exp(log_prec_2 - log_prec_1) + np.power(mean_1-mean_2,2)/var_2 )
    # when log_prec_2 > -np.inf, log_prec_1=-np.inf leads to infinite kl
    if len_1 == 1:
        cond = np.logical_and(np.isinf(log_prec_1), log_prec_1 < 0)
        idx_log_prec_1_neginf = np.ones(len_max) * cond
        if cond:
            kl = np.inf * np.ones(len_max)
    else:
        idx_log_prec_1_neginf = np.logical_and(np.isinf(log_prec_1), log_prec_1 < 0)
        kl[idx_log_prec_1_neginf] = np.inf
    if len_2 == 1:
        cond = np.logical_and(np.isinf(log_prec_2), log_prec_2 < 0)
        idx_log_prec_2_neginf = np.ones(len_max) * cond
    else:
        idx_log_prec_2_neginf = np.logical_and(np.isinf(log_prec_2), log_prec_2 < 0)
    # when log_prec_2 = -np.inf, log_prec_1=-np.inf leads to zero kl
    idx_both_log_prec_neginf = np.logical_and(idx_log_prec_1_neginf, idx_log_prec_2_neginf)
    kl[idx_both_log_prec_neginf] = 0.
    # when log_prec_2 = np.inf, any value of log_prec_1 leads to infinite kl
    idx_log_prec_2_posinf = np.logical_and(np.isinf(log_prec_2), log_prec_2 > 0)
    if (len_2 == 1) and idx_log_prec_2_posinf:
        kl = np.inf * np.ones(len_max)
    else:
        kl[idx_log_prec_2_posinf] = np.inf
    if False:
        print 'log_prec_1 = %s, log_prec_2 = %s, kl = %s' % (log_prec_1, log_prec_2, kl)
    if np.any(np.isnan(kl)):
        print '\nsomething went wrong with kl computation'
        print 'var_1 = %s, var_2 = %s' % (var_1, var_2)
        print 'log_prec_1 = %s, log_prec_2 = %s' % (log_prec_1, log_prec_2)
        print 'idx_log_prec_1_neginf = %s' % idx_log_prec_1_neginf
        print 'idx_log_prec_2_neginf = %s' % idx_log_prec_2_neginf
        print 'idx_log_prec_2_posinf = %s' % idx_log_prec_2_posinf
        print 'kl = %s' % kl
        raise Exception
    return kl


def test_compute_kl():
    compute_kl(0*np.ones(2), np.inf*np.ones(2), 0*np.ones(1), np.inf*np.ones(1))
    compute_kl(0*np.ones(2), -np.inf*np.ones(2), 0*np.ones(1), np.inf*np.ones(1))
    compute_kl(0*np.ones(2), np.inf*np.ones(2), 0*np.ones(1), -np.inf*np.ones(1))
    compute_kl(0*np.ones(2), -np.inf*np.ones(2), 0*np.ones(1), -np.inf*np.ones(1))
    compute_kl(0*np.ones(1), np.inf*np.ones(1), 0*np.ones(2), np.inf*np.ones(2))
    compute_kl(0*np.ones(1), -np.inf*np.ones(1), 0*np.ones(2), np.inf*np.ones(2))
    compute_kl(0*np.ones(1), np.inf*np.ones(1), 0*np.ones(2), -np.inf*np.ones(2))
    compute_kl(0*np.ones(1), -np.inf*np.ones(1), 0*np.ones(2), -np.inf*np.ones(2))


def multiply_gaussians(*params):
    """
    input is a list containing (variable number of) gaussian parameters
    each element is a numpy array containing mean and precision of that gaussian
    """
    precision_op, mean_op = 0., 0.
    for param in params:
        precision_op += param[1]
        mean_op += param[0] * param[1]
    mean_op /= precision_op
    return np.array([mean_op, precision_op])


def divide_gaussians(mean_precision_num, mean_precision_den):
    """
    mean_precision_num are parameters of gaussian in the numerator
    mean_precision_den are parameters of gaussian in the denominator
    output is a valid gaussian only if the variance of ouput is non-negative
    """
    precision_op = mean_precision_num[1] - mean_precision_den[1]
    try:
        assert precision_op >= 0.        #   making it > so that mean_op is not inf
    except AssertionError:
        print 'inputs = %s, %s' % (mean_precision_num, mean_precision_den)
        print 'precision_op = %s' % (precision_op)
        raise AssertionError
    if precision_op == 0.:
        mean_op = 0.
    else:
        mean_op = (mean_precision_num[0] * mean_precision_num[1] \
                     - mean_precision_den[0] * mean_precision_den[1] ) / precision_op
    return np.array([mean_op, precision_op])


def hist_count(x, basis):
    """
    counts number of times each element in basis appears in x
    op is a vector of same size as basis
    assume no duplicates in basis
    """
    op = np.zeros((len(basis)), dtype=int)
    map_basis = {}
    for n, k in enumerate(basis):
        map_basis[k] = n
    for t in x:
        op[map_basis[t]] += 1
    return op


def logsumexp(x):
    tmp = x.copy()
    tmp_max = np.max(tmp)
    tmp -= tmp_max
    op = np.log(np.sum(np.exp(tmp))) + tmp_max
    return op


def logsumexp_array(v1, v2):
    """
    computes logsumexp of each element in v1 and v2
    """
    v_min = np.minimum(v1, v2)
    v_max = np.maximum(v1, v2)
    op = v_max + np.log(1 + np.exp(v_min - v_max))
    return op


def logsumexp_2(x, y):
    # fast logsumexp for 2 variables
    # output = log (e^x + e^y) = log(e^max(1+e^(min-max))) = max + log(1 + e^(min-max))
    if x > y:
        min_val = y
        max_val = x
    else:
        min_val = x
        max_val = y
    op = max_val + math.log(1 + math.exp(min_val - max_val))
    return op


def softmax(x):
    tmp = x.copy()
    tmp_max = np.max(tmp)
    tmp -= float(tmp_max)
    tmp = np.exp(tmp)
    op = tmp / np.sum(tmp)
    return op


def assert_no_nan(mat, name='matrix'):
    try:
        assert(not any(np.isnan(mat)))
    except AssertionError:
        print '%s contains NaN' % name
        print mat
        raise AssertionError

def check_if_one(val):
    try:
        assert(np.abs(val - 1) < 1e-9)
    except AssertionError:
        print 'val = %s (needs to be equal to 1)' % val
        raise AssertionError

def check_if_zero(val):
    try:
        assert(np.abs(val) < 1e-9)
    except AssertionError:
        print 'val = %s (needs to be equal to 0)' % val
        raise AssertionError


def linear_regression(x, y):
    ls = np.linalg.lstsq(x, y)
    #print ls
    coef = ls[0]
    if ls[1]:
        sum_squared_residuals = float(ls[1])    # sum of squared residuals
    else:
        sum_squared_residuals = np.sum(np.dot(x, coef) - y)    # sum of squared residuals
    return (coef, sum_squared_residuals)


def sample_multinomial(prob):
    try:
        k = int(np.where(np.random.multinomial(1, prob, size=1)[0]==1)[0])
    except TypeError:
        print 'problem in sample_multinomial: prob = '
        print prob
        raise TypeError
    except:
        raise Exception
    return k


def sample_multinomial_scores(scores):
    scores_cumsum = np.cumsum(scores)
    s = scores_cumsum[-1] * np.random.rand(1)
    k = int(np.sum(s > scores_cumsum))
    return k


def sample_multinomial_scores_old(scores):
    scores_cumsum = np.cumsum(scores)
    s = scores_cumsum[-1] * np.random.rand(1)
    k = 0
    while s > scores_cumsum[k]:
        k += 1
    return k


def sample_polya(alpha_vec, n):
    """ alpha_vec is the parameter of the Dirichlet distribution, n is the #samples """
    prob = np.random.dirichlet(alpha_vec)
    n_vec = np.random.multinomial(n, prob)
    return n_vec


def get_kth_minimum(x, k=1):
    """ gets the k^th minimum element of the list x 
        (note: k=1 is the minimum, k=2 is 2nd minimum) ...
        based on the incomplete selection sort pseudocode """
    n = len(x)
    for i in range(n):
        minIndex = i
        minValue = x[i]
        for j in range(i+1, n):
            if x[j] < minValue:
                minIndex = j
                minValue = x[j]
        x[i], x[minIndex] = x[minIndex], x[i]
    return x[k-1]


class empty(object):
    def __init__(self):
        pass


def sigmoid(x):
    op = 1.0 / (1 + np.exp(-x))
    return op


def compute_m_sd(x):
    m = np.mean(x)
    s = np.sqrt(np.var(x))
    return (m, s)

if __name__ == "__main__":
    test_compute_kl()
