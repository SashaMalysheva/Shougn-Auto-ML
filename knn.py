import os
from scipy.io import loadmat
from numpy import random
from modshogun import CrossValidationSplitting
import time
from modshogun import MulticlassLabels, RealFeatures
from modshogun import KNN, EuclideanDistance, ManhattanMetric
from modshogun import MulticlassAccuracy
from modshogun import KNN_COVER_TREE, KNN_BRUTE
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import numpy as np
SHOGUN_DATA_DIR = os.getenv('SHOGUN_DATA_DIR', '/home/sasha/workspace/shogun/data')


def get_data():
    """
    Get a random subset of 1000 samples from the USPS digit recognition dataset.
    """
    mat = loadmat(os.path.join(SHOGUN_DATA_DIR, 'multiclass/usps.mat'))
    Xall = mat['data']
    Yall = np.array(mat['label'].squeeze(), dtype=np.double) - 1
    subset = random.permutation(len(Yall))
    # Xtrain, Ytrain, Xtest, Ytest
    return Xall[:, subset[:500]], Yall[subset[:500]], Xall[:, subset[500:600]], Yall[subset[500:600]]


def generate_labels(Xtrain, Ytrain, Xtest, Ytest):
    labels = MulticlassLabels(Ytrain)
    feats = RealFeatures(Xtrain)
    labels_test = MulticlassLabels(Ytest)
    feats_test = RealFeatures(Xtest)
    return labels, feats, labels_test, feats_test


Xtrain, Ytrain, Xtest, Ytest = get_data()
labels, feats, labels_test, feats_test = generate_labels(Xtrain, Ytrain, Xtest, Ytest)


def evaluate(labels,
             feats,
             params={'n_neighbors': 2, 'use_cover_tree': 'True', 'dist': 'Manhattan'},
             Nsplit=2):
    """
        Run Cross-validation to evaluate the KNN.

        Parameters
        ----------
        labels: 2d array
            Data set labels.
        feats: array
            Data set feats.
        params: dictionary
            Search scope parameters.
        Nsplit: int, default = 2
            The n for n-fold cross validation.
        all_ks: range of int, default = range(1, 21)
            Numbers of neighbors.
    """
    k = params.get('n_neighbors')
    use_cover_tree = params.get('use_cover_tree') == 'True'
    if params.get('dist' == 'Euclidean'):
        func_dist = EuclideanDistance
    else:
        func_dist = ManhattanMetric

    split = CrossValidationSplitting(labels, Nsplit)
    split.build_subsets()

    accuracy = np.zeros(Nsplit)
    acc_train = np.zeros(accuracy.shape)
    time_test = np.zeros(accuracy.shape)
    for i in range(Nsplit):
        idx_train = split.generate_subset_inverse(i)
        idx_test = split.generate_subset_indices(i)

        feats.add_subset(idx_train)
        labels.add_subset(idx_train)

        dist = func_dist(feats, feats)
        knn = KNN(k, dist, labels)
        knn.set_store_model_features(True)
        if use_cover_tree:
            knn.set_knn_solver_type(KNN_COVER_TREE)
        else:
            knn.set_knn_solver_type(KNN_BRUTE)
        knn.train()

        evaluator = MulticlassAccuracy()
        pred = knn.apply_multiclass()
        acc_train[i] = evaluator.evaluate(pred, labels)

        feats.remove_subset()
        labels.remove_subset()
        feats.add_subset(idx_test)
        labels.add_subset(idx_test)

        t_start = time.clock()
        pred = knn.apply_multiclass(feats)
        time_test[i] = (time.clock() - t_start) / labels.get_num_labels()

        accuracy[i] = evaluator.evaluate(pred, labels)

        feats.remove_subset()
        labels.remove_subset()
    print accuracy.mean()
    return accuracy


def hyperopt_train_test(params):
    return evaluate(labels, feats, params).mean()


##############################
##==== Global variables ====##
##############################
"""
Define the search space.

Parameters
----------
n_neighbors: int in range(1, 20)
    Number of neighbors.
use_cover_tree: boolean
    The flag of enable or disable this Cover Trees.
dist: string 'Euclidean' or 'Manhattan'
    The distance metric to use for the tree.
"""
space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(2, 20)),
    'use_cover_tree': hp.choice('use_cover_tree', ['True', 'False']),
    'dist': hp.choice('dist', ['Euclidean', 'Manhattan'])
}


# Define minimized function
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=10, trials=trials)
print('best:')
print(best)

# Draw plot of the function.
parameters = ['n_neighbors', 'use_cover_tree', 'dist']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(15, 5))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    xs, ys = zip(*sorted(zip(xs, ys)))
    ys = np.array(ys)
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i) / len(parameters)))
    axes[i].set_title(val)
plt.show()
