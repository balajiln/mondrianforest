#!/usr/bin/env python
#
# Example usage:
#
# NOTE:
# optype=real: Gaussian parametrization uses a non-linear transformation of split times
#   variance should decrease as split_time increases: 
#   variance at node j = variance_coef * (sigmoid(sigmoid_coef * t_j) - sigmoid(sigmoid_coef * t_{parent(j)}))
#   non-linear transformation should be a monotonically non-decreasing function
#   sigmoid has a saturation effect: children will be similar to parent as we go down the tree
#   split times t_j scales inversely with the number of dimensions

import sys
import os
import optparse
import math
import time
import cPickle as pickle
import random
import pprint as pp
import numpy as np
from warnings import warn
from utils import hist_count, logsumexp, softmax, sample_multinomial, \
        sample_multinomial_scores, empty, assert_no_nan, check_if_zero, check_if_one, \
        multiply_gaussians, divide_gaussians, sigmoid, logsumexp_array
from mondrianforest_utils import Forest, Param, parser_add_common_options, parser_check_common_options, \
        bootstrap, parser_add_mf_options, parser_check_mf_options, reset_random_seed, \
        load_data, add_stuff_2_settings, compute_gaussian_pdf, compute_gaussian_logpdf, \
        get_filename_mf, precompute_minimal, compute_left_right_statistics, \
        create_prediction_tree, init_prediction_tree, update_predictive_posterior_node, \
        compute_metrics_classification, compute_metrics_regression, \
        update_posterior_node_incremental, init_update_posterior_node_incremental
from itertools import izip, count, chain
from collections import defaultdict
try:
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib import rc
    rc('font', **{'family':'serif'})
    rc('text', usetex=True)
    rc('legend', handlelength=4)
    rc('legend', **{'fontsize':9})
except:
    warn('matplotlib not loaded: plotting not possible; set draw_mondrian=0')
try:
    import pydot
except:
    warn('pydot not loaded: tree will not be printed; set draw_mondrian=0')
# setting numpy options to debug RuntimeWarnings
#np.seterr(divide='raise')
np.seterr(divide='ignore')      # to avoid warnings for np.log(0)
np.seterr(invalid='ignore')      # to avoid warnings for inf * 0 = nan
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=200)
# color scheme for mondrian
# colors_list = ['DarkRed', 'Navy', 'DimGray', 'Beige']  
# other nice colors: Beige, MediumBlue, DarkRed vs FireBrick
colors_list = ['LightGray']  # paused leaf will always be shaded gray
LW = 2
FIGSIZE = (12, 9)
INF = np.inf


def process_command_line():
    parser = parser_add_common_options()
    parser = parser_add_mf_options(parser)
    settings, args = parser.parse_args()
    add_stuff_2_settings(settings)
    if settings.optype == 'class':
        settings.alpha = 0    # normalized stable prior
        assert settings.smooth_hierarchically
    parser_check_common_options(parser, settings)
    parser_check_mf_options(parser, settings)
    if settings.budget < 0:
        settings.budget_to_use = INF
    else:
        settings.budget_to_use = settings.budget
    return settings


class MondrianBlock(object):
    """ 
    defines Mondrian block
    variables:
    - min_d         : dimension-wise min of training data in current block
    - max_d         : dimension-wise max of training data in current block 
    - range_d       : max_d - min_d
    - sum_range_d   : sum of range_d
    - left          : id of left child
    - right         : id of right child
    - parent        : id of parent
    - is_leaf       : boolen variable to indicate if current block is leaf
    - budget        : remaining lifetime for subtree rooted at current block
                      = lifetime of Mondrian - time of split of parent 
                      NOTE: time of split of parent of root node is 0 
    """
    def __init__(self, data, settings, budget, parent, range_stats):
        self.min_d, self.max_d, self.range_d, self.sum_range_d = range_stats
        self.budget = budget + 0.
        self.parent = parent
        self.left = None
        self.right = None
        self.is_leaf = True


class MondrianTree(object):
    """
    defines a Mondrian tree
    variables:
    - node_info     : stores splits for internal nodes
    - root          : id of root node
    - leaf_nodes    : list of leaf nodes
    - non_leaf_nodes: list of non-leaf nodes
    - max_split_costs   : max_split_cost for a node is time of split of node - time of split of parent
                          max_split_cost is drawn from an exponential
    - train_ids     : list of train ids stored for paused Mondrian blocks
    - counts        : stores histogram of labels at each node (when optype = 'class')
    - grow_nodes    : list of Mondrian blocks that need to be "grown"
    functions:
    - __init__      : initialize a Mondrian tree
    - grow          : samples Mondrian block (more precisely, restriction of blocks to training data)
    - extend_mondrian   : extend a Mondrian to include new training data
    - extend_mondrian_block : conditional Mondrian algorithm
    """
    def __init__(self, data=None, train_ids=None, settings=None, param=None, cache=None):
        """
        initialize Mondrian tree data structure and sample restriction of Mondrian tree to current training data
        data is a N x D numpy array containing the entire training data
        train_ids is the training ids of the first minibatch
        """
        if data is None:
            return
        root_node = MondrianBlock(data, settings, settings.budget_to_use, None, \
                        get_data_range(data, train_ids))
        self.root = root_node
        self.non_leaf_nodes = []
        self.leaf_nodes = []
        self.node_info = {}
        self.max_split_costs = {}
        self.split_times = {}
        self.train_ids = {root_node: train_ids}
        self.copy_params(param, settings)
        init_prediction_tree(self, settings)
        if cache:
            if settings.optype == 'class':
                self.counts = {root_node: cache['y_train_counts']}
            else:
                self.sum_y = {root_node: cache['sum_y']}
                self.sum_y2 = {root_node: cache['sum_y2']}
                self.n_points = {root_node: cache['n_points']}
            if settings.bagging == 1 or settings.n_minibatches > 1:
                init_update_posterior_node_incremental(self, data, param, settings, cache, root_node, train_ids)
        self.grow_nodes = [root_node]
        self.grow(data, settings, param, cache)

    def copy_params(self, param, settings):
        if settings.optype == 'real':
            self.noise_variance = param.noise_variance + 0 
            self.noise_precision = param.noise_precision + 0
            self.sigmoid_coef = param.sigmoid_coef + 0 
            self.variance_coef = param.variance_coef + 0

    def get_average_depth(self, settings, data):
        """
        compute average depth of tree (averaged over training data)
        = depth of a leaf weighted by fraction of training data at that leaf
        """
        self.depth_nodes = {self.root: 0}
        tmp_node_list = [self.root]
        n_total = 0.
        average_depth = 0.
        self.node_size_by_depth = defaultdict(list)
        leaf_node_sizes = []
        while True:
            try:
                node_id = tmp_node_list.pop(0)
            except IndexError:
                break
            if node_id.is_leaf:
                if settings.optype == 'class':
                    n_points_node = np.sum(self.counts[node_id])
                else:
                    n_points_node = self.n_points[node_id]
                n_total += n_points_node
                average_depth += n_points_node * self.depth_nodes[node_id]
                self.node_size_by_depth[self.depth_nodes[node_id]].append(node_id.sum_range_d)
            if not node_id.is_leaf:
                self.depth_nodes[node_id.left] = self.depth_nodes[node_id] + 1
                self.depth_nodes[node_id.right] = self.depth_nodes[node_id] + 1
                tmp_node_list.extend([node_id.left, node_id.right])
            else:
                leaf_node_sizes.append(node_id.sum_range_d)
        assert data['n_train'] == int(n_total)
        average_depth /= n_total
        average_leaf_node_size = np.mean(leaf_node_sizes)
        average_node_size_by_depth = {}
        for k in self.node_size_by_depth:
            average_node_size_by_depth[k] = np.mean(self.node_size_by_depth[k])
        return (average_depth, average_leaf_node_size, average_node_size_by_depth)

    def get_print_label_draw_tree(self, node_id, graph):
        """
        helper function for draw_tree using pydot
        """
        name = self.node_ids_print[node_id]
        name2 = name
        if name2 == '':
            name2 = 'e'
        if node_id.is_leaf:
            op = name
        else:
            feat_id, split = self.node_info[node_id]
            op = r'x_%d > %.2f\nt = %.2f' % (feat_id+1, split, self.cumulative_split_costs[node_id])
        if op == '':
            op = 'e'
        node = pydot.Node(name=name2, label=op) # latex labels don't work
        graph.add_node(node)
        return (name2, graph)

    def draw_tree(self, data, settings, figure_id=0, i_t=0):
        """
        function to draw Mondrian tree using pydot
        NOTE: set ADD_TIME=True if you want want set edge length between parent and child 
                to the difference in time of splits
        """
        self.gen_node_ids_print()
        self.gen_cumulative_split_costs_only(settings, data)
        graph = pydot.Dot(graph_type='digraph')
        dummy, graph = self.get_print_label_draw_tree(self.root, graph)
        ADD_TIME = False
        for node_id in self.non_leaf_nodes:
            parent, graph = self.get_print_label_draw_tree(node_id, graph)
            left, graph = self.get_print_label_draw_tree(node_id.left, graph)
            right, graph = self.get_print_label_draw_tree(node_id.right, graph)
            for child, child_id in izip([left, right], [node_id.left, node_id.right]):
                edge = pydot.Edge(parent, child)
                if ADD_TIME and (not child_id.is_leaf):
                    edge.set_minlen(self.max_split_costs[child_id])
                    edge2 = pydot.Edge(dummy, child)
                    edge2.set_minlen(self.cumulative_split_costs[child_id])
                    edge2.set_style('invis')
                    graph.add_edge(edge2)
                graph.add_edge(edge)
        filename_plot_tag = get_filename_mf(settings)[:-2]
        if settings.save:
            tree_name = filename_plot_tag + '-mtree_minibatch-' + str(figure_id) + '.pdf'
            print 'saving file: %s' % tree_name
            graph.write_pdf(tree_name)
        
    def draw_mondrian(self, data, settings, figure_id=None, i_t=0):
        """ 
        function to draw Mondrian partitions; each Mondrian tree is one subplot.
        """
        assert data['n_dim'] == 2 and settings.normalize_features == 1 \
                and settings.n_mondrians <= 10
        self.gen_node_list()
        if settings.n_mondrians == 1 and settings.dataset == 'toy-mf':
            self.draw_tree(data, settings, figure_id, i_t)
        if settings.n_mondrians > 2:
            n_row = 2
        else:
            n_row = 1
        n_col = int(math.ceil(settings.n_mondrians / n_row))
        if figure_id is None:
            figure_id = 0
        fig = plt.figure(figure_id)
        plt.hold(True)
        ax = plt.subplot(n_row, n_col, i_t+1, aspect='equal')
        EPS = 0.
        ax.set_xlim(xmin=0-EPS)
        ax.set_xlim(xmax=1+EPS)
        ax.set_ylim(ymin=0-EPS)
        ax.set_ylim(ymax=1+EPS)
        ax.autoscale(False)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        non_leaf_nodes = [self.root]
        while non_leaf_nodes:
            node_id = non_leaf_nodes.pop(0)
            try:
                feat_id, split = self.node_info[node_id]
            except:
                continue
            left, right = node_id.left, node_id.right
            non_leaf_nodes.append(left)
            non_leaf_nodes.append(right)
            EXTRA = 0.0    # to show splits that separate 2 data points
            if feat_id == 1:
                # axhline doesn't work if you rescale
                ax.hlines(split, node_id.min_d[0] - EXTRA, node_id.max_d[0] + EXTRA, lw=LW, color='k')
            else:
                ax.vlines(split, node_id.min_d[1] - EXTRA, node_id.max_d[1] + EXTRA, lw=LW, color='k')
        # add "outer patch" that defines the extent (not data dependent)
        block = patches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='gray', ls='dashed')
        ax.add_patch(block)
        for i_, node_id in enumerate(self.node_list):
            # plot only the block where Mondrian has been induced (limited by extent of training data)
            block = patches.Rectangle((node_id.min_d[0], node_id.min_d[1]), node_id.range_d[0], \
                    node_id.range_d[1], facecolor='white', edgecolor='gray')
            ax.add_patch(block)
        for i_, node_id in enumerate(self.leaf_nodes):
            # plot only the block where Mondrian has been induced (limited by extent of training data)
            block = patches.Rectangle((node_id.min_d[0], node_id.min_d[1]), node_id.range_d[0], \
                    node_id.range_d[1], facecolor=colors_list[i_ % len(colors_list)], edgecolor='black')
            ax.add_patch(block)
            # zorder = 1 will make points inside the blocks invisible, >= 2 will make them visible
            x_train = data['x_train'][self.train_ids[node_id], :]
            #ax.scatter(x_train[:, 0], x_train[:, 1], color='k', marker='x', s=10, zorder=2)
            color_y = 'rbk'
            for y_ in range(data['n_class']):
                idx = data['y_train'][self.train_ids[node_id]] == y_
                ax.scatter(x_train[idx, 0], x_train[idx, 1], color=color_y[y_], marker='o', s=16, zorder=2)
        plt.draw()

    def gen_node_ids_print(self):
        """
        generate binary string label for each node
        root_node is denoted by empty string "e"
        all other node labels are defined as follows: left(j) = j0, right(j) = j1
        e.g. left and right child of root_node are 0 and 1 respectively, 
             left and right of node 0 are 00 and 01 respectively and so on.
        """
        node_ids = [self.root]
        self.node_ids_print = {self.root: ''}
        while node_ids:
            node_id = node_ids.pop(0)
            try:
                feat_id, split = self.node_info[node_id]
                left, right = node_id.left, node_id.right
                node_ids.append(left)
                node_ids.append(right)
                self.node_ids_print[left] = self.node_ids_print[node_id] + '0'
                self.node_ids_print[right] = self.node_ids_print[node_id] + '1'
            except KeyError:
                continue

    def print_dict(self, d):
        """
        print a dictionary
        """
        for k in d:
            print '\tk_map = %10s, val = %s' % (self.node_ids_print[k], d[k])

    def print_list(self, list_):
        """
        print a list
        """
        print '\t%s' % ([self.node_ids_print[x] for x in list_])

    def print_tree(self, settings):
        """
        prints some tree statistics: leaf nodes, non-leaf nodes, information and so on
        """
        self.gen_node_ids_print()
        print 'printing tree:'
        print 'len(leaf_nodes) = %s, len(non_leaf_nodes) = %s' \
                % (len(self.leaf_nodes), len(self.non_leaf_nodes))
        print 'node_info ='
        node_ids = [self.root]
        while node_ids:
            node_id = node_ids.pop(0)
            node_id_print = self.node_ids_print[node_id]
            try:
                feat_id, split = self.node_info[node_id]
                print '%10s, feat = %5d, split = %.2f, node_id = %s' % \
                        (node_id_print, feat_id, split, node_id)
                if settings.optype == 'class':
                    print 'counts = %s' % self.counts[node_id]
                else:
                    print 'n_points = %6d, sum_y = %.2f' % (self.n_points[node_id], self.sum_y[node_id])
                left, right = node_id.left, node_id.right
                node_ids.append(left)
                node_ids.append(right)
            except KeyError:
                continue
        print 'leaf info ='
        for node_id in self.leaf_nodes:
            node_id_print = self.node_ids_print[node_id]
            print '%10s, train_ids = %s, node_id = %s' % \
                    (node_id_print, self.train_ids[node_id], node_id)
            if settings.optype == 'class':
                print 'counts = %s' % self.counts[node_id]
            else:
                print 'n_points = %6d, sum_y = %.2f' % (self.n_points[node_id], self.sum_y[node_id])

    def check_if_labels_same(self, node_id):
        """
        checks if all labels in a node are identical
        """
        return np.count_nonzero(self.counts[node_id]) == 1

    def pause_mondrian(self, node_id, settings):
        """
        should you pause a Mondrian block or not?
        pause if sum_range_d == 0 (important for handling duplicates) or
        - optype == class: pause if all labels in a node are identical 
        - optype == real: pause if n_points < min_samples_split
        """
        if settings.optype == 'class':
            pause_mondrian_tmp = self.check_if_labels_same(node_id)
        else:
            pause_mondrian_tmp = self.n_points[node_id] < settings.min_samples_split
        pause_mondrian = pause_mondrian_tmp or (node_id.sum_range_d == 0)
        return pause_mondrian

    def get_parent_split_time(self, node_id, settings):
        if node_id == self.root:
            return 0.
        else:
            return self.split_times[node_id.parent]
    
    def update_gaussian_hyperparameters(self, param, data, settings):
        n_points = float(self.n_points[self.root])
        param.prior_mean = self.sum_y[self.root] / n_points
        param.prior_variance = self.sum_y2[self.root] / n_points \
                                - param.prior_mean ** 2
        param.prior_precision = 1.0 / param.prior_variance
        # TODO: estimate K using estimate of noise variance at leaf nodes?
        # TODO: need to do this once for forest, rather than for each tree
        # FIXME very very hacky, surely a better way to tune this?
        if 'sfactor' in settings.tag:
            s_begin = settings.tag.find('sfactor-') + 8
            s_tmp = settings.tag[s_begin:]
            s_factor = float(s_tmp[:s_tmp.find('-')])
        else:
            s_factor = 2.0
        if 'kfactor' in settings.tag:
            k_begin = settings.tag.find('kfactor-') + 8
            k_tmp = settings.tag[k_begin:]
            k_factor = float(k_tmp[:k_tmp.find('-')])
        else:
            k_factor = min(2 * n_points, 500)  # noise variance is 1/K times prior_variance
        if k_factor <= 0.:
            K = 2. * n_points
        else:
            K = k_factor
        param.noise_variance = param.prior_variance / K
        param.noise_precision = 1.0 / param.noise_variance
        param.variance_coef = 2.0 * param.prior_variance * K / (K + 2.)
        param.sigmoid_coef = data['n_dim']  / (s_factor * np.log2(n_points))
        # FIXME: important to copy over since prediction accesses hyperparameters in self
        self.copy_params(param, settings)

    def get_node_mean_and_variance(self, node):
        n_points = float(self.n_points[node])
        node_mean = self.sum_y[node] / n_points
        node_variance = self.sum_y2[node] / n_points - node_mean ** 2
        return (node_mean, node_variance)

    def update_gaussian_hyperparameters_indep(self, param, data, settings):
        n_points = float(self.n_points[self.root])
        self.prior_mean, self.prior_variance = self.get_node_mean_and_variance(self.root)
        self.prior_precision = 1.0 / self.prior_variance
        self.cumulative_split_costs = {}
        self.leaf_means = []
        self.leaf_variances = []
        node_means = []
        d_node_means = {self.root: self.prior_mean}
        node_parent_means = []
        node_split_times = []
        node_parent_split_times = []
        if self.root.is_leaf:
            self.cumulative_split_costs[self.root] = 0.
            remaining = []
            self.max_split_time = 0.1   # NOTE: initial value, need to specify non-zero value
        else:
            self.cumulative_split_costs[self.root] = self.max_split_costs[self.root]
            remaining = [self.root.left, self.root.right]
            self.max_split_time = self.cumulative_split_costs[self.root] + 0
            node_split_times.append(self.cumulative_split_costs[self.root])
            node_parent_split_times.append(0.)
            node_means.append(self.prior_mean)
            node_parent_means.append(self.prior_mean)
        while True:
            try:
                node_id = remaining.pop(0)
            except IndexError:
                break
            self.cumulative_split_costs[node_id] = self.cumulative_split_costs[node_id.parent] \
                                                    + self.max_split_costs[node_id]
            node_mean, node_variance = self.get_node_mean_and_variance(node_id)
            node_split_times.append(self.cumulative_split_costs[node_id])
            node_parent_split_times.append(self.cumulative_split_costs[node_id.parent])
            node_means.append(node_mean)
            node_parent_means.append(d_node_means[node_id.parent])
            d_node_means[node_id] = node_mean
            if not node_id.is_leaf:
                remaining.append(node_id.left)
                remaining.append(node_id.right)
                self.max_split_time = max(self.max_split_time, self.cumulative_split_costs[node_id])
            else:
                self.leaf_means.append(node_mean)
                self.leaf_variances.append(node_variance)
        #self.noise_variance = np.max(self.leaf_variances)
        self.noise_variance = np.mean(self.leaf_variances)
        self.noise_precision = 1.0 / self.noise_variance
        self.sigmoid_coef = 3. / self.max_split_time
        #self.sigmoid_coef = data['n_dim']
        #self.sigmoid_coef = data['n_dim'] / 5
        #self.sigmoid_coef = data['n_dim']  / (2. * np.log2(n_points))
        #self.sigmoid_coef = data['n_dim']  / (2. * np.log2(n_points))
        #self.sigmoid_coef = data['n_dim']  / (n_points)
        #self.variance_leaf_from_root = 2 * np.mean((np.array(self.leaf_means) - self.prior_mean) ** 2)
        # set sd to 3 times the empirical sd so that leaf node means are highly plausible (avoid too much smoothing)
        #self.variance_coef = 1.0 * self.variance_leaf_from_root     
        if self.root.is_leaf:
            self.variance_coef = 1.0
        else:
            node_means = np.array(node_means)
            node_parent_means = np.array(node_parent_means)
            node_split_times = np.array(node_split_times)
            node_parent_split_times = np.array(node_parent_split_times)
            tmp_den = sigmoid(self.sigmoid_coef * node_split_times) \
                        - sigmoid(self.sigmoid_coef * node_parent_split_times)
            tmp_num = (node_means - node_parent_means) ** 2
            variance_coef_est = np.mean(tmp_num / tmp_den)
            self.variance_coef = variance_coef_est
            print 'sigmoid_coef = %.3f, variance_coef = %.3f' % (self.sigmoid_coef, variance_coef_est)

    def grow(self, data, settings, param, cache):
        """
        sample a Mondrian tree (each Mondrian block is restricted to range of training data in that block)
        """
        if settings.debug:
            print 'entering grow'
        while self.grow_nodes:
            node_id = self.grow_nodes.pop(0)
            train_ids = self.train_ids[node_id]
            if settings.debug:
                print 'node_id = %s' % node_id
            pause_mondrian = self.pause_mondrian(node_id, settings)
            if settings.debug and pause_mondrian:
                print 'pausing mondrian at node = %s, train_ids = %s' % (node_id, self.train_ids[node_id])
            if pause_mondrian or (node_id.sum_range_d == 0):    # BL: redundant now
                split_cost = np.inf
                self.max_split_costs[node_id] = node_id.budget + 0
                self.split_times[node_id] = np.inf  # FIXME: is this correct? inf or budget?
            else:
                split_cost = random.expovariate(node_id.sum_range_d)
                self.max_split_costs[node_id] = split_cost
                self.split_times[node_id] = split_cost + self.get_parent_split_time(node_id, settings)
            new_budget = node_id.budget - split_cost
            if node_id.budget > split_cost:
                feat_id_chosen = sample_multinomial_scores(node_id.range_d)
                split_chosen = random.uniform(node_id.min_d[feat_id_chosen], \
                                                node_id.max_d[feat_id_chosen])
                (train_ids_left, train_ids_right, cache_tmp) = \
                    compute_left_right_statistics(data, param, cache, train_ids, feat_id_chosen, split_chosen, settings)
                left = MondrianBlock(data, settings, new_budget, node_id, get_data_range(data, train_ids_left))
                right = MondrianBlock(data, settings, new_budget, node_id, get_data_range(data, train_ids_right))
                node_id.left, node_id.right = left, right
                self.grow_nodes.append(left)
                self.grow_nodes.append(right)
                self.train_ids[left] = train_ids_left
                self.train_ids[right] = train_ids_right
                if settings.optype == 'class':
                    self.counts[left] = cache_tmp['cnt_left_chosen']
                    self.counts[right] = cache_tmp['cnt_right_chosen']
                else:
                    self.sum_y[left] = cache_tmp['sum_y_left']
                    self.sum_y2[left] = cache_tmp['sum_y2_left']
                    self.n_points[left] = cache_tmp['n_points_left']
                    self.sum_y[right] = cache_tmp['sum_y_right']
                    self.sum_y2[right] = cache_tmp['sum_y2_right']
                    self.n_points[right] = cache_tmp['n_points_right']
                self.node_info[node_id] = [feat_id_chosen, split_chosen]
                self.non_leaf_nodes.append(node_id)
                node_id.is_leaf = False
                if not settings.draw_mondrian:
                    self.train_ids.pop(node_id)
            else:
                self.leaf_nodes.append(node_id)     # node_id.is_leaf set to True at init

    def gen_cumulative_split_costs_only(self, settings, data):
        """
        creates node_id.cumulative_split_cost as well as a dictionary self.cumulative_split_costs
        helper function for draw_tree
        """
        self.cumulative_split_costs = {}
        if self.root.is_leaf:
            self.cumulative_split_costs[self.root] = 0.
            remaining = []
        else:
            self.cumulative_split_costs[self.root] = self.max_split_costs[self.root]
            remaining = [self.root.left, self.root.right]
        while True:
            try:
                node_id = remaining.pop(0)
            except IndexError:
                break
            self.cumulative_split_costs[node_id] = self.cumulative_split_costs[node_id.parent] \
                                                    + self.max_split_costs[node_id]
            if not node_id.is_leaf:
                remaining.append(node_id.left)
                remaining.append(node_id.right)

    def gen_node_list(self):
        """
        generates an ordered node_list such that parent appears before children
        useful for updating predictive posteriors
        """
        self.node_list = [self.root]
        i = -1
        while True:
            try:
                i += 1
                node_id = self.node_list[i]
            except IndexError:
                break
            if not node_id.is_leaf:
                self.node_list.extend([node_id.left, node_id.right])

    def predict_class(self, x_test, n_class, param, settings):
        """
        predict new label (for classification tasks)
        """
        pred_prob = np.zeros((x_test.shape[0], n_class))
        prob_not_separated_yet = np.ones(x_test.shape[0])
        prob_separated = np.zeros(x_test.shape[0])
        node_list = [self.root]
        d_idx_test = {self.root: np.arange(x_test.shape[0])}
        while True:
            try:
                node_id = node_list.pop(0)
            except IndexError:
                break
            idx_test = d_idx_test[node_id]
            if len(idx_test) == 0:
                continue
            x = x_test[idx_test, :]
            expo_parameter = np.maximum(0, node_id.min_d - x).sum(1) + np.maximum(0, x - node_id.max_d).sum(1)
            prob_not_separated_now = np.exp(-expo_parameter * self.max_split_costs[node_id])
            prob_separated_now = 1 - prob_not_separated_now
            if math.isinf(self.max_split_costs[node_id]):
                # rare scenario where test point overlaps exactly with a training data point
                idx_zero = expo_parameter == 0
                # to prevent nan in computation above when test point overlaps with training data point
                prob_not_separated_now[idx_zero] = 1.
                prob_separated_now[idx_zero] = 0.
            # predictions for idx_test_zero
            # data dependent discounting (depending on how far test data point is from the mondrian block)
            idx_non_zero = expo_parameter > 0
            idx_test_non_zero = idx_test[idx_non_zero]
            expo_parameter_non_zero = expo_parameter[idx_non_zero]
            base = self.get_prior_mean(node_id, param, settings)
            if np.any(idx_non_zero):
                num_tables_k, num_customers, num_tables = self.get_counts(self.cnt[node_id])
                # expected discount (averaging over time of cut which is a truncated exponential)
                # discount = (expo_parameter_non_zero / (expo_parameter_non_zero + settings.discount_param)) * \
                #       (-np.expm1(-(expo_parameter_non_zero + settings.discount_param) * self.max_split_costs[node_id]))
                discount = (expo_parameter_non_zero / (expo_parameter_non_zero + settings.discount_param)) \
                    * (-np.expm1(-(expo_parameter_non_zero + settings.discount_param) * self.max_split_costs[node_id])) \
                    / (-np.expm1(-expo_parameter_non_zero * self.max_split_costs[node_id]))
                discount_per_num_customers = discount / num_customers
                pred_prob_tmp = num_tables * discount_per_num_customers[:, np.newaxis] * base \
                        + self.cnt[node_id] / num_customers - discount_per_num_customers[:, np.newaxis] * num_tables_k
                pred_prob[idx_test_non_zero, :] += prob_separated_now[idx_non_zero][:, np.newaxis] \
                                            * prob_not_separated_yet[idx_test_non_zero][:, np.newaxis] * pred_prob_tmp
                prob_not_separated_yet[idx_test] *= prob_not_separated_now
            # predictions for idx_test_zero
            if math.isinf(self.max_split_costs[node_id]) and np.any(idx_zero):
                idx_test_zero = idx_test[idx_zero]
                pred_prob_node_id = self.compute_posterior_mean_normalized_stable(self.cnt[node_id], \
                                            self.get_discount_node_id(node_id, settings), base, settings)
                pred_prob[idx_test_zero, :] += prob_not_separated_yet[idx_test_zero][:, np.newaxis] * pred_prob_node_id
            try:
                feat_id, split = self.node_info[node_id]
                cond = x[:, feat_id] <= split
                left, right = get_children_id(node_id)
                d_idx_test[left], d_idx_test[right] = idx_test[cond], idx_test[~cond]
                node_list.append(left)
                node_list.append(right)
            except KeyError:
                pass
        if True or settings.debug: 
            check_if_zero(np.sum(np.abs(np.sum(pred_prob, 1) - 1)))
        return pred_prob

    def predict_real(self, x_test, y_test, param, settings):
        """
        predict new label (for regression tasks)
        """
        pred_mean = np.zeros(x_test.shape[0])
        pred_second_moment = np.zeros(x_test.shape[0])
        pred_sample = np.zeros(x_test.shape[0])
        log_pred_prob = -np.inf * np.ones(x_test.shape[0])
        prob_not_separated_yet = np.ones(x_test.shape[0])
        prob_separated = np.zeros(x_test.shape[0])
        node_list = [self.root]
        d_idx_test = {self.root: np.arange(x_test.shape[0])}
        while True:
            try:
                node_id = node_list.pop(0)
            except IndexError:
                break
            idx_test = d_idx_test[node_id]
            if len(idx_test) == 0:
                continue
            x = x_test[idx_test, :]
            expo_parameter = np.maximum(0, node_id.min_d - x).sum(1) + np.maximum(0, x - node_id.max_d).sum(1)
            prob_not_separated_now = np.exp(-expo_parameter * self.max_split_costs[node_id])
            prob_separated_now = 1 - prob_not_separated_now
            if math.isinf(self.max_split_costs[node_id]):
                # rare scenario where test point overlaps exactly with a training data point
                idx_zero = expo_parameter == 0
                # to prevent nan in computation above when test point overlaps with training data point
                prob_not_separated_now[idx_zero] = 1.
                prob_separated_now[idx_zero] = 0.
            # predictions for idx_test_zero
            idx_non_zero = expo_parameter > 0
            idx_test_non_zero = idx_test[idx_non_zero]
            n_test_non_zero = len(idx_test_non_zero)
            expo_parameter_non_zero = expo_parameter[idx_non_zero]
            if np.any(idx_non_zero):
                # expected variance (averaging over time of cut which is a truncated exponential)
                # NOTE: expected variance is approximate since E[f(x)] not equal to f(E[x])
                expected_cut_time = 1.0 / expo_parameter_non_zero
                if not np.isinf(self.max_split_costs[node_id]):
                    tmp_exp_term_arg = -self.max_split_costs[node_id] * expo_parameter_non_zero
                    tmp_exp_term = np.exp(tmp_exp_term_arg)
                    expected_cut_time -= self.max_split_costs[node_id] * tmp_exp_term / (-np.expm1(tmp_exp_term_arg))
                try:
                    assert np.all(expected_cut_time >= 0.)
                except AssertionError:
                    print tmp_exp_term_arg
                    print tmp_exp_term
                    print expected_cut_time
                    print np.any(np.isnan(expected_cut_time))
                    print 1.0 / expo_parameter_non_zero
                    raise AssertionError
                if not settings.smooth_hierarchically:
                    pred_mean_tmp = self.sum_y[node_id] / float(self.n_points[node_id])
                    pred_second_moment_tmp = self.sum_y2[node_id] / float(self.n_points[node_id]) + param.noise_variance
                else:
                    pred_mean_tmp, pred_second_moment_tmp = self.pred_moments[node_id]
                    # FIXME: approximate since E[f(x)] not equal to f(E[x])
                    expected_split_time = expected_cut_time + self.get_parent_split_time(node_id, settings)
                    variance_from_mean = self.variance_coef * (sigmoid(self.sigmoid_coef * expected_split_time) \
                                        - sigmoid(self.sigmoid_coef * self.get_parent_split_time(node_id, settings)))
                    pred_second_moment_tmp += variance_from_mean
                pred_variance_tmp = pred_second_moment_tmp - pred_mean_tmp ** 2
                pred_sample_tmp = pred_mean_tmp + np.random.randn(n_test_non_zero) * np.sqrt(pred_variance_tmp)
                log_pred_prob_tmp = compute_gaussian_logpdf(pred_mean_tmp, pred_variance_tmp, y_test[idx_test_non_zero])
                prob_separated_now_weighted = \
                        prob_separated_now[idx_non_zero] * prob_not_separated_yet[idx_test_non_zero]
                pred_mean[idx_test_non_zero] += prob_separated_now_weighted * pred_mean_tmp
                pred_sample[idx_test_non_zero] += prob_separated_now_weighted * pred_sample_tmp
                pred_second_moment[idx_test_non_zero] += prob_separated_now_weighted * pred_second_moment_tmp
                log_pred_prob[idx_test_non_zero] = logsumexp_array(log_pred_prob[idx_test_non_zero], \
                                                    np.log(prob_separated_now_weighted) + log_pred_prob_tmp)
                prob_not_separated_yet[idx_test] *= prob_not_separated_now
            # predictions for idx_test_zero
            if math.isinf(self.max_split_costs[node_id]) and np.any(idx_zero):
                idx_test_zero = idx_test[idx_zero]
                n_test_zero = len(idx_test_zero)
                if not settings.smooth_hierarchically:
                    pred_mean_node_id = self.sum_y[node_id] / float(self.n_points[node_id])
                    pred_second_moment_node_id = self.sum_y2[node_id] / float(self.n_points[node_id]) \
                                                    + param.noise_variance
                else:
                    pred_mean_node_id, pred_second_moment_node_id = self.pred_moments[node_id]
                pred_variance_node_id = pred_second_moment_node_id - pred_mean_node_id ** 2
                pred_sample_node_id = pred_mean_node_id + np.random.randn(n_test_zero) * np.sqrt(pred_variance_node_id)
                log_pred_prob_node_id = \
                        compute_gaussian_logpdf(pred_mean_node_id, pred_variance_node_id, y_test[idx_test_zero])
                pred_mean[idx_test_zero] += prob_not_separated_yet[idx_test_zero] * pred_mean_node_id
                pred_sample[idx_test_zero] += prob_not_separated_yet[idx_test_zero] * pred_sample_node_id
                pred_second_moment[idx_test_zero] += prob_not_separated_yet[idx_test_zero] * pred_second_moment_node_id
                log_pred_prob[idx_test_zero] = logsumexp_array(log_pred_prob[idx_test_zero], \
                                                np.log(prob_not_separated_yet[idx_test_zero]) + log_pred_prob_node_id)
            try:
                feat_id, split = self.node_info[node_id]
                cond = x[:, feat_id] <= split
                left, right = get_children_id(node_id)
                d_idx_test[left], d_idx_test[right] = idx_test[cond], idx_test[~cond]
                node_list.append(left)
                node_list.append(right)
            except KeyError:
                pass
        pred_var = pred_second_moment - (pred_mean ** 2)
        if True or settings.debug:  # FIXME: remove later
            assert not np.any(np.isnan(pred_mean))
            assert not np.any(np.isnan(pred_var))
            try:
                assert np.all(pred_var >= 0.)
            except AssertionError:
                min_pred_var = np.min(pred_var)
                print 'min_pred_var = %s' % min_pred_var
                assert np.abs(min_pred_var) < 1e-3  # allowing some numerical errors
            assert not np.any(np.isnan(log_pred_prob))
        return (pred_mean, pred_var, pred_second_moment, log_pred_prob, pred_sample)

    def extend_mondrian(self, data, train_ids_new, settings, param, cache):
        """
        extend Mondrian tree to include new training data indexed by train_ids_new
        """
        self.extend_mondrian_block(self.root, train_ids_new, data, settings, param, cache)
        if settings.debug:
            print 'completed extend_mondrian'
            self.check_tree(settings, data)

    def check_tree(self, settings, data):
        """
        check if tree violates any sanity check
        """
        if settings.debug:
            #print '\nchecking tree'
            print '\nchecking tree: printing tree first'
            self.print_tree(settings)
        for node_id in self.non_leaf_nodes:
            assert node_id.left.parent == node_id.right.parent == node_id
            assert not node_id.is_leaf
            if settings.optype == 'class':
                assert np.count_nonzero(self.counts[node_id]) > 1
            assert not self.pause_mondrian(node_id, settings)
            if node_id != self.root:
                assert np.all(node_id.min_d >= node_id.parent.min_d)
                assert np.all(node_id.max_d <= node_id.parent.max_d)
            if settings.optype == 'class':
                try:
                    check_if_zero(np.sum(np.abs(self.counts[node_id] - \
                            self.counts[node_id.left] - self.counts[node_id.right])))
                except AssertionError:
                    print 'counts: node = %s, left = %s, right = %s' \
                            % (self.counts[node_id], self.counts[node_id.left], self.counts[node_id.right])
                    raise AssertionError
            if settings.budget == -1:
                assert math.isinf(node_id.budget)
            check_if_zero(self.split_times[node_id] - self.get_parent_split_time(node_id, settings) \
                    - self.max_split_costs[node_id])
        if settings.optype == 'class':
            num_data_points = 0
        for node_id in self.leaf_nodes:
            assert node_id.is_leaf
            assert math.isinf(self.max_split_costs[node_id])
            if settings.budget == -1:
                assert math.isinf(node_id.budget)
            if settings.optype == 'class':
                num_data_points += self.counts[node_id].sum()
                assert np.count_nonzero(self.counts[node_id]) == 1
                assert self.pause_mondrian(node_id, settings)
            if node_id != self.root:
                assert np.all(node_id.min_d >= node_id.parent.min_d)
                assert np.all(node_id.max_d <= node_id.parent.max_d)
        if settings.optype == 'class':
            print 'num_train = %s, number of data points at leaf nodes = %s' % \
                    (data['n_train'], num_data_points)
        set_non_leaf = set(self.non_leaf_nodes)
        set_leaf = set(self.leaf_nodes)
        assert (set_leaf & set_non_leaf) == set([])
        assert set_non_leaf == set(self.node_info.keys())
        assert len(set_leaf) == len(self.leaf_nodes)
        assert len(set_non_leaf) == len(self.non_leaf_nodes)

    def extend_mondrian_block(self, node_id, train_ids_new, data, settings, param, cache):
        """
        conditional Mondrian algorithm that extends a Mondrian block to include new training data
        """
        if settings.debug:
            print 'entered extend_mondrian_block'
            print '\nextend_mondrian_block: node_id = %s' % node_id
        if not train_ids_new.size:
            if settings.debug:
                print 'nothing to extend here; train_ids_new = %s' % train_ids_new
            # nothing to extend
            return
        min_d, max_d = get_data_min_max(data, train_ids_new)
        additional_extent_lower = np.maximum(0, node_id.min_d - min_d)
        additional_extent_upper = np.maximum(0, max_d - node_id.max_d)
        expo_parameter = float(additional_extent_lower.sum() + additional_extent_upper.sum())
        if expo_parameter == 0:
            split_cost = np.inf
        else:
            split_cost = random.expovariate(expo_parameter)     # will be updated below in case mondrian is paused
        unpause_paused_mondrian = False
        if settings.debug:
            print 'is_leaf = %s, pause_mondrian = %s, sum_range_d = %s' % \
                    (node_id.is_leaf, self.pause_mondrian(node_id, settings), node_id.sum_range_d)
        if self.pause_mondrian(node_id, settings):
            assert node_id.is_leaf
            split_cost = np.inf
            if settings.optype == 'class':
                y_unique = np.unique(data['y_train'][train_ids_new])
                # FIXME: node_id.sum_range_d not tested
                unpause_paused_mondrian = not( (len(y_unique) == 1) and (self.counts[node_id][y_unique] > 0) )
            else:
                n_points_new = len(data['y_train'][train_ids_new])
                unpause_paused_mondrian = \
                        not( (n_points_new + self.n_points[node_id]) < settings.min_samples_split )
                        # node_id.sum_range_d not tested
            if settings.debug:
                print 'trying to extend a paused Mondrian; is_leaf = %s, node_id = %s' % (node_id.is_leaf, node_id)
                if settings.optype == 'class':
                    print 'y_unique = %s, counts = %s, split_cost = %s, max_split_costs = %s' % \
                        (y_unique, self.counts[node_id], split_cost, self.max_split_costs[node_id])
        if split_cost >= self.max_split_costs[node_id]:
            # take root form of node_id (no cut outside the extent of the current block)
            if not node_id.is_leaf:
                if settings.debug:
                    print 'take root form: non-leaf node'
                feat_id, split = self.node_info[node_id]
                update_range_stats(node_id, (min_d, max_d)) # required here as well
                left, right = node_id.left, node_id.right
                cond = data['x_train'][train_ids_new, feat_id] <= split
                train_ids_new_left, train_ids_new_right = train_ids_new[cond], train_ids_new[~cond]
                self.add_training_points_to_node(node_id, train_ids_new, data, param, settings, cache, False)
                self.extend_mondrian_block(left, train_ids_new_left, data, settings, param, cache)
                self.extend_mondrian_block(right, train_ids_new_right, data, settings, param, cache)
            else:
                # reached a leaf; add train_ids_new to node_id & update range
                if settings.debug:
                    print 'take root form: leaf node'
                assert node_id.is_leaf
                update_range_stats(node_id, (min_d, max_d))
                self.add_training_points_to_node(node_id, train_ids_new, data, param, settings, cache, True)
                # FIXME: node_id.sum_range_d tested here; perhaps move this to pause_mondrian?
                unpause_paused_mondrian = unpause_paused_mondrian and (node_id.sum_range_d != 0)
                if not self.pause_mondrian(node_id, settings):
                    assert unpause_paused_mondrian
                    self.leaf_nodes.remove(node_id)
                    self.grow_nodes = [node_id]
                    self.grow(data, settings, param, cache)
        else:
            # initialize "outer mondrian"
            if settings.debug:
                print 'trying to introduce a cut outside current block'
            new_block = MondrianBlock(data, settings, node_id.budget, node_id.parent, \
                        get_data_range_from_min_max(np.minimum(min_d, node_id.min_d), np.maximum(max_d, node_id.max_d)))
            init_update_posterior_node_incremental(self, data, param, settings, cache, new_block, \
                    train_ids_new, node_id)      # counts of outer block are initialized with counts of current block
            if node_id.is_leaf:
                warn('\nWARNING: a leaf should not be expanded here; printing out some diagnostics')
                print 'node_id = %s, is_leaf = %s, max_split_cost = %s, split_cost = %s' \
                        % (node_id, node_id.is_leaf, self.max_split_costs[node_id], split_cost)
                print 'counts = %s\nmin_d = \n%s\nmax_d = \n%s' % (self.counts[node_id], node_id.min_d, node_id.max_d)
                raise Exception('a leaf should be expanded via grow call; see diagnostics above')
            if settings.debug:
                print 'looks like cut possible'
            # there is a cut outside the extent of the current block
            feat_score = additional_extent_lower + additional_extent_upper
            feat_id = sample_multinomial_scores(feat_score)
            draw_from_lower = np.random.rand() <= (additional_extent_lower[feat_id] / feat_score[feat_id])
            if draw_from_lower:
                split = random.uniform(min_d[feat_id], node_id.min_d[feat_id])
            else:
                split = random.uniform(node_id.max_d[feat_id], max_d[feat_id])
            assert (split < node_id.min_d[feat_id]) or (split > node_id.max_d[feat_id])
            new_budget = node_id.budget - split_cost
            cond = data['x_train'][train_ids_new, feat_id] <= split
            train_ids_new_left, train_ids_new_right = train_ids_new[cond], train_ids_new[~cond]
            is_left = split > node_id.max_d[feat_id]    # is existing block the left child of "outer mondrian"?
            if is_left:
                train_ids_new_child = train_ids_new_right   # new_child is the other child of "outer mondrian"
            else:
                train_ids_new_child = train_ids_new_left
            # grow the "unconditional mondrian child" of the "outer mondrian"
            new_child = MondrianBlock(data, settings, new_budget, new_block, get_data_range(data, train_ids_new_child))
            if settings.debug:
                print 'new_block = %s' % new_block
                print 'new_child = %s' % new_child
            self.train_ids[new_child] = train_ids_new_child     # required for grow call below
            init_update_posterior_node_incremental(self, data, param, settings, cache, new_child, train_ids_new_child)
            self.node_info[new_block] = (feat_id, split)
            if settings.draw_mondrian:
                train_ids_new_block = np.append(self.train_ids[node_id], train_ids_new)
                self.train_ids[new_block] = train_ids_new_block
            self.non_leaf_nodes.append(new_block)
            new_block.is_leaf = False
            # update budget and call the "conditional mondrian child" of the "outer mondrian"
            node_id.budget = new_budget
            # self.max_split_costs[new_child] will be added in the grow call above
            self.max_split_costs[new_block] = split_cost
            self.split_times[new_block] = split_cost + self.get_parent_split_time(node_id, settings)
            self.max_split_costs[node_id] -= split_cost
            check_if_zero(self.split_times[node_id] - self.split_times[new_block] - self.max_split_costs[node_id])
            # grow the new child of the "outer mondrian"
            self.grow_nodes = [new_child]
            self.grow(data, settings, param, cache)
            # update tree structure and extend "conditional mondrian child" of the "outer mondrian"
            if node_id == self.root:
                self.root = new_block
            else:
                if settings.debug:
                    assert (node_id.parent.left == node_id) or (node_id.parent.right == node_id)
                if node_id.parent.left == node_id:
                    node_id.parent.left = new_block
                else:
                    node_id.parent.right = new_block
            node_id.parent = new_block
            if is_left:
                new_block.left = node_id
                new_block.right = new_child
                self.extend_mondrian_block(node_id, train_ids_new_left, data, settings, param, cache)
            else:
                new_block.left = new_child
                new_block.right = node_id
                self.extend_mondrian_block(node_id, train_ids_new_right, data, settings, param, cache)

    def add_training_points_to_node(self, node_id, train_ids_new, data, param, settings, cache, pause_mondrian=False):
        """
        add a training data point to a node in the tree
        """
        # range updated in extend_mondrian_block
        if settings.draw_mondrian or pause_mondrian:
            self.train_ids[node_id] = np.append(self.train_ids[node_id], train_ids_new)
        update_posterior_node_incremental(self, data, param, settings, cache, node_id, train_ids_new)

    def update_posterior_counts(self, param, data, settings):
        """
        posterior update for hierarchical normalized stable distribution
        using interpolated Kneser Ney smoothing (where number of tables serving a dish at a restaurant is atmost 1)
        NOTE: implementation optimized for minibatch training where more than one data point added per minibatch
        if only 1 datapoint is added, lots of counts will be unnecesarily updated
        """
        self.cnt = {}
        node_list = [self.root]
        while True:
            try:
                node_id = node_list.pop(0)
            except IndexError:
                break
            if node_id.is_leaf:
                cnt = self.counts[node_id]
            else:
                cnt = np.minimum(self.counts[node_id.left], 1) + np.minimum(self.counts[node_id.right], 1)
                node_list.extend([node_id.left, node_id.right])
            self.cnt[node_id] = cnt

    def update_predictive_posteriors(self, param, data, settings):
        """
        update predictive posterior for hierarchical normalized stable distribution
        pred_prob computes posterior mean of the label distribution at each node recursively
        """
        node_list = [self.root]
        if settings.debug:
            self.gen_node_ids_print()
        while True:
            try:
                node_id = node_list.pop(0)
            except IndexError:
                break
            base = self.get_prior_mean(node_id, param, settings)
            discount = self.get_discount_node_id(node_id, settings)
            cnt = self.cnt[node_id]
            if not node_id.is_leaf:
                self.pred_prob[node_id] = self.compute_posterior_mean_normalized_stable(cnt, discount, base, settings)
                node_list.extend([node_id.left, node_id.right])
            if settings.debug and False:
                print 'node_id = %20s, is_leaf = %5s, discount = %.2f, cnt = %s, base = %s, pred_prob = %s' \
                        % (self.node_ids_print[node_id], node_id.is_leaf, discount, cnt, base, self.pred_prob[node_id])

    def get_variance_node(self, node_id, param, settings):
        # the non-linear transformation should be a monotonically non-decreasing function
        # if the function saturates (e.g. sigmoid) children will be closer to parent deeper down the tree
        # var = self.variance_coef * (sigmoid(self.sigmoid_coef * self.split_times[node_id]) \
        #        - sigmoid(self.sigmoid_coef * self.get_parent_split_time(node_id, settings)))
        var = self.variance_coef * (sigmoid(self.sigmoid_coef * self.split_times[node_id]) \
                - sigmoid(self.sigmoid_coef * self.get_parent_split_time(node_id, settings)))
        return var

    def update_posterior_gaussians(self, param, data, settings):
        """
        computes marginal gaussian distribution at each node of the tree using gaussian belief propagation
        the solution is exact since underlying graph is a tree
        solution takes O(#nodes) time, which is much more efficient than naive GP implementation which 
        would cost O(#nodes^3) time
        """
        self.gen_node_list()
        self.message_to_parent = {}
        self.message_from_parent = {}
        self.likelihood_children = {}
        self.pred_param = {}
        self.pred_moments = {}
        for node_id in self.node_list[::-1]:
            if node_id.is_leaf:
                # use marginal likelihood of data at this leaf
                mean = self.sum_y[node_id] / float(self.n_points[node_id])
                variance = self.get_variance_node(node_id, param, settings) \
                            + self.noise_variance / float(self.n_points[node_id])
                precision = 1.0 / variance
                self.message_to_parent[node_id] = np.array([mean, precision])
                self.likelihood_children[node_id] = np.array([mean, self.noise_precision*float(self.n_points[node_id])])
            else:
                likelihood_children = multiply_gaussians(self.message_to_parent[node_id.left], \
                                                    self.message_to_parent[node_id.right])
                mean = likelihood_children[0]
                self.likelihood_children[node_id] = likelihood_children
                variance = self.get_variance_node(node_id, param, settings) + 1.0 / likelihood_children[1]
                precision = 1.0 / variance
                self.message_to_parent[node_id] = np.array([mean, precision])
        variance_at_root = self.get_variance_node(node_id, param, settings)
        self.message_from_parent[self.root] = np.array([param.prior_mean, variance_at_root])
        for node_id in self.node_list:
            # pred_param stores the mean and precision
            self.pred_param[node_id] = multiply_gaussians(self.message_from_parent[node_id], \
                                            self.likelihood_children[node_id])
            # pred_moments stores the first and second moments (useful for prediction)
            self.pred_moments[node_id] = np.array([self.pred_param[node_id][0], \
                    1.0 / self.pred_param[node_id][1] + self.pred_param[node_id][0] ** 2 + self.noise_variance])
            if not node_id.is_leaf:
                self.message_from_parent[node_id.left] = \
                        multiply_gaussians(self.message_from_parent[node_id], self.message_to_parent[node_id.right])
                self.message_from_parent[node_id.right] = \
                        multiply_gaussians(self.message_from_parent[node_id], self.message_to_parent[node_id.left])

    def update_posterior_counts_and_predictive_posteriors(self, param, data, settings):
        if settings.optype == 'class':
            # update posterior counts
            self.update_posterior_counts(param, data, settings)
            # update predictive posteriors
            self.update_predictive_posteriors(param, data, settings)
        else:
            # updates hyperparameters in param (common to all trees)
            self.update_gaussian_hyperparameters(param, data, settings)
            # updates hyperparameters in self (independent for each tree)
            # self.update_gaussian_hyperparameters_indep(param, data, settings)
            if settings.smooth_hierarchically:
                self.update_posterior_gaussians(param, data, settings)

    def get_prior_mean(self, node_id, param, settings):
        if settings.optype == 'class':
            if node_id == self.root:
                base = param.base_measure
            else:
                base = self.pred_prob[node_id.parent]
        else:
            base = None     # for settings.settings.smooth_hierarchically = False
        return base

    def get_discount_node_id(self, node_id, settings):
        """
        compute discount for a node (function of discount_param, time of split and time of split of parent)
        """
        discount = math.exp(-settings.discount_param * self.max_split_costs[node_id])
        return discount

    def compute_posterior_mean_normalized_stable(self, cnt, discount, base, settings):
        num_tables_k, num_customers, num_tables = self.get_counts(cnt)
        pred_prob = (cnt - discount * num_tables_k + discount * num_tables * base) / num_customers
        if settings.debug:
            check_if_one(pred_prob.sum())
        return pred_prob

    def get_counts(self, cnt):
        num_tables_k = np.minimum(cnt, 1)
        num_customers = float(cnt.sum())
        num_tables = float(num_tables_k.sum())
        return (num_tables_k, num_customers, num_tables)


def get_data_range(data, train_ids):
    """
    returns min, max, range and linear dimension of training data
    """
    min_d, max_d = get_data_min_max(data, train_ids)
    range_d = max_d - min_d
    sum_range_d = float(range_d.sum())
    return (min_d, max_d, range_d, sum_range_d)


def get_data_min_max(data, train_ids):
    """
    returns min, max of training data
    """
    x_tmp = data['x_train'].take(train_ids, 0)
    min_d = np.min(x_tmp, 0)
    max_d = np.max(x_tmp, 0)
    return (min_d, max_d)


def get_data_range_from_min_max(min_d, max_d):
    range_d = max_d - min_d
    sum_range_d = float(range_d.sum())
    return (min_d, max_d, range_d, sum_range_d)


def update_range_stats(node_id, (min_d, max_d)):
    """
    updates min and max of training data at this block
    """
    node_id.min_d = np.minimum(node_id.min_d, min_d)
    node_id.max_d = np.maximum(node_id.max_d, max_d)
    node_id.range_d = node_id.max_d - node_id.min_d
    node_id.sum_range_d = float(node_id.range_d.sum())
    

def get_children_id(parent):
    return (parent.left, parent.right)


class MondrianForest(Forest):
    """ 
    defines Mondrian forest
    variables:
    - forest     : stores the Mondrian forest
    methods:
    - fit(data, train_ids_current_minibatch, settings, param, cache)            : batch training
    - partial_fit(data, train_ids_current_minibatch, settings, param, cache)    : online training
    - evaluate_predictions (see Forest in mondrianforest_utils.py)              : predictions
    """
    def __init__(self, settings, data):
        self.forest = [None] * settings.n_mondrians
        if settings.optype == 'class':
            settings.discount_param = settings.discount_factor * data['n_dim']

    def fit(self, data, train_ids_current_minibatch, settings, param, cache):
        for i_t, tree in enumerate(self.forest):
            if settings.verbose >= 2 or settings.debug:
                print 'tree_id = %s' % i_t
            tree = self.forest[i_t] = MondrianTree(data, train_ids_current_minibatch, settings, param, cache)
            tree.update_posterior_counts_and_predictive_posteriors(param, data, settings)

    def partial_fit(self, data, train_ids_current_minibatch, settings, param, cache):
        for i_t, tree in enumerate(self.forest):
            if settings.verbose >= 2 or settings.debug:
                print 'tree_id = %s' % i_t
            tree.extend_mondrian(data, train_ids_current_minibatch, settings, param, cache)
            tree.update_posterior_counts_and_predictive_posteriors(param, data, settings)

def main():
    time_0 = time.clock()
    settings = process_command_line()
    print
    print '%' * 120
    print 'Beginning mondrianforest.py'
    print 'Current settings:'
    pp.pprint(vars(settings))

    # Resetting random seed
    reset_random_seed(settings)

    # Loading data
    print '\nLoading data ...'
    data = load_data(settings)
    print 'Loading data ... completed'
    print 'Dataset name = %s' % settings.dataset
    print 'Characteristics of the dataset:'
    print 'n_train = %d, n_test = %d, n_dim = %d' %\
            (data['n_train'], data['n_test'], data['n_dim'])
    if settings.optype == 'class':
        print 'n_class = %d' % (data['n_class'])

    # precomputation
    param, cache = precompute_minimal(data, settings)
    time_init = time.clock() - time_0

    print '\nCreating Mondrian forest'
    # online training with minibatches
    time_method_sans_init = 0.
    time_prediction = 0.
    mf = MondrianForest(settings, data)
    if settings.store_every:
        log_prob_test_minibatch = -np.inf * np.ones(settings.n_minibatches)
        log_prob_train_minibatch = -np.inf * np.ones(settings.n_minibatches)
        metric_test_minibatch = -np.inf * np.ones(settings.n_minibatches)
        metric_train_minibatch = -np.inf * np.ones(settings.n_minibatches)
        time_method_minibatch = np.inf * np.ones(settings.n_minibatches)
        forest_numleaves_minibatch = np.zeros(settings.n_minibatches)
    for idx_minibatch in range(settings.n_minibatches):
        time_method_init = time.clock()
        is_last_minibatch = (idx_minibatch == settings.n_minibatches - 1)
        print_results = is_last_minibatch or (settings.verbose >= 2) or settings.debug
        if print_results:
            print '*' * 120
            print 'idx_minibatch = %5d' % idx_minibatch
        train_ids_current_minibatch = data['train_ids_partition']['current'][idx_minibatch]
        if settings.debug:
            print 'bagging = %s, train_ids_current_minibatch = %s' % \
                    (settings.bagging, train_ids_current_minibatch)
        if idx_minibatch == 0:
            mf.fit(data, train_ids_current_minibatch, settings, param, cache)
        else:
            mf.partial_fit(data, train_ids_current_minibatch, settings, param, cache)
        for i_t, tree in enumerate(mf.forest):
            if settings.debug or settings.verbose >= 2:
                print '-'*100
                tree.print_tree(settings)
                print '.'*100
            if settings.draw_mondrian:
                tree.draw_mondrian(data, settings, idx_minibatch, i_t)
                if settings.save == 1:
                    filename_plot = get_filename_mf(settings)[:-2]
                    if settings.store_every:
                        plt.savefig(filename_plot + '-mondrians_minibatch-' + str(idx_minibatch) + '.pdf', format='pdf')
        time_method_sans_init += time.clock() - time_method_init
        time_method = time_method_sans_init + time_init

        # Evaluate
        if is_last_minibatch or settings.store_every:
            time_predictions_init = time.clock()
            weights_prediction = np.ones(settings.n_mondrians) * 1.0 / settings.n_mondrians
            if False:
                if print_results:
                    print 'Results on training data (log predictive prob is bogus)'
                train_ids_cumulative = data['train_ids_partition']['cumulative'][idx_minibatch]
                # NOTE: some of these data points are not used for "training" if bagging is used
                pred_forest_train, metrics_train = \
                        mf.evaluate_predictions(data, data['x_train'][train_ids_cumulative, :], \
                        data['y_train'][train_ids_cumulative], \
                        settings, param, weights_prediction, print_results)
            else:
                # not computing metrics on training data
                metrics_train = {'log_prob': -np.inf, 'acc': 0, 'mse': np.inf}
                pred_forest_train = None
            if print_results:
                print '\nResults on test data'
            pred_forest_test, metrics_test = \
                mf.evaluate_predictions(data, data['x_test'], data['y_test'], \
                settings, param, weights_prediction, print_results)
            name_metric = settings.name_metric     # acc or mse
            log_prob_train = metrics_train['log_prob']
            log_prob_test = metrics_test['log_prob']
            metric_train = metrics_train[name_metric]
            metric_test = metrics_test[name_metric]
            if settings.store_every:
                log_prob_train_minibatch[idx_minibatch] = metrics_train['log_prob']
                log_prob_test_minibatch[idx_minibatch] = metrics_test['log_prob']
                metric_train_minibatch[idx_minibatch] = metrics_train[name_metric]
                metric_test_minibatch[idx_minibatch] = metrics_test[name_metric]
                time_method_minibatch[idx_minibatch] = time_method
                tree_numleaves = np.zeros(settings.n_mondrians)
                for i_t, tree in enumerate(mf.forest):
                    tree_numleaves[i_t] = len(tree.leaf_nodes)
                forest_numleaves_minibatch[idx_minibatch] = np.mean(tree_numleaves)
            time_prediction += time.clock() - time_predictions_init

    # printing test performance:
    if settings.store_every:
        print 'printing test performance for every minibatch:'
        print 'idx_minibatch\tmetric_test\ttime_method\tnum_leaves'
        for idx_minibatch in range(settings.n_minibatches):
            print '%10d\t%.3f\t\t%.3f\t\t%.1f' % \
                    (idx_minibatch, \
                    metric_test_minibatch[idx_minibatch], \
                    time_method_minibatch[idx_minibatch], forest_numleaves_minibatch[idx_minibatch])
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

    if settings.draw_mondrian:
        if settings.save == 1:
            plt.savefig(filename_plot + '-mondrians-final.pdf', format='pdf')
        else:
            plt.show()

    # Write results to disk (timing doesn't include saving)
    time_total = time.clock() - time_0
    # resetting
    if settings.save == 1:
        filename = get_filename_mf(settings)
        print 'filename = ' + filename
        results = {'log_prob_test': log_prob_test, 'log_prob_train': log_prob_train, \
                    'metric_test': metric_test, 'metric_train': metric_train, \
                'time_total': time_total, 'time_method': time_method, \
                'time_init': time_init, 'time_method_sans_init': time_method_sans_init,\
                'time_prediction': time_prediction}
        if 'log_prob2' in metrics_test:
            results['log_prob2_test'] = metrics_test['log_prob2']
        store_data = settings.dataset[:3] == 'toy' or settings.dataset == 'sim-reg'
        if store_data:
            results['data'] = data
        if settings.store_every:
            results['log_prob_test_minibatch'] = log_prob_test_minibatch
            results['log_prob_train_minibatch'] = log_prob_train_minibatch
            results['metric_test_minibatch'] = metric_test_minibatch
            results['metric_train_minibatch'] = metric_train_minibatch
            results['time_method_minibatch'] = time_method_minibatch
            results['forest_numleaves_minibatch'] = forest_numleaves_minibatch
        results['settings'] = settings
        results['tree_stats'] = tree_stats
        results['tree_average_depth'] = tree_average_depth
        pickle.dump(results, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        # storing final predictions as well; recreate new "results" dict
        results = {'pred_forest_train': pred_forest_train, \
                    'pred_forest_test': pred_forest_test}
        filename2 = filename[:-2] + '.tree_predictions.p'
        pickle.dump(results, open(filename2, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    
    time_total = time.clock() - time_0
    print
    print 'Time for initializing Mondrian forest (seconds) = %f' % (time_init)
    print 'Time for executing mondrianforest.py (seconds) = %f' % (time_method_sans_init)
    print 'Total time for executing mondrianforest.py, including init (seconds) = %f' % (time_method)
    print 'Time for prediction/evaluation (seconds) = %f' % (time_prediction)
    print 'Total time (Loading data/ initializing / running / predictions / saving) (seconds) = %f\n' % (time_total)

if __name__ == "__main__":
    main()
