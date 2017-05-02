import os
from modshogun import GaussianKernel, GMNPSVM, LinearKernel, SigmoidKernel
from scipy.io import loadmat
from numpy import random
from modshogun import CrossValidationSplitting
import time
from modshogun import MulticlassLabels, RealFeatures
from modshogun import KNN, EuclideanDistance, ManhattanMetric
from modshogun import MulticlassAccuracy
from modshogun import KNN_COVER_TREE, KNN_BRUTE
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
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


def evaluate4knn(labels,
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
    return accuracy


def evaluate4svm(labels,
                 feats,
                 params={'c': 1, 'kernal': 'gauss'},
                 Nsplit=2):
    """
        Run Cross-validation to evaluate the SVM.

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
    """
    c = params.get('c')
    if params.get('kernal' == 'gauss'):
        kernal = GaussianKernel()
        kernal.set_width(80)
    elif params.get('kernal' == 'sigmoid'):
        kernal = SigmoidKernel()
    else:
        kernal = LinearKernel()

    split = CrossValidationSplitting(labels, Nsplit)
    split.build_subsets()

    accuracy = np.zeros(Nsplit)
    time_test = np.zeros(accuracy.shape)
    for i in range(Nsplit):
        idx_train = split.generate_subset_inverse(i)
        idx_test = split.generate_subset_indices(i)

        feats.add_subset(idx_train)
        labels.add_subset(idx_train)
        svm = GMNPSVM(c, kernal, labels)
        _ = svm.train(feats)
        out = svm.apply(feats_test)
        evaluator = MulticlassAccuracy()
        accuracy[i] = evaluator.evaluate(out, labels_test)

        feats.remove_subset()
        labels.remove_subset()
        feats.add_subset(idx_test)
        labels.add_subset(idx_test)

        t_start = time.clock()
        time_test[i] = (time.clock() - t_start) / labels.get_num_labels()
        feats.remove_subset()
        labels.remove_subset()
    return accuracy


def hyperopt_train_test(params):
    t = params['type']
    del params['type']
    if t == 'svm':
        acc = evaluate4svm(labels, feats, params).mean()
    elif t == 'knn':
        acc = evaluate4knn(labels, feats, params).mean()
    else:
        return 0
    return acc


##############################
##==== Global variables ====##
##############################
"""
Define the search space.

Parameters
----------
type == 'svm'
    C: int in range(0, 3)
        Penalty parameter of the error term.
    kernel: string - 'linear' or 'gauss' or 'sigmoid'
        Specifies the kernel type to be used in the algorithm.
type == 'knn'
    n_neighbors: int in range(1, 20)
        Number of neighbors.
    use_cover_tree: boolean
        The flag of enable or disable this Cover Trees.
    dist: string 'Euclidean' or 'Manhattan'
        The distance metric to use for the tree.
"""
space = hp.choice('classifier_type', [
    {
        'type': 'svm',
        'c': hp.choice('c', range(0, 3)),
        'kernal': hp.choice('kernel', ['linear', 'gauss', 'sigmoid'])
    },
    {
        'type': 'knn',
        'n_neighbors': hp.choice('n_neighbors', range(2, 20)),
        'use_cover_tree': hp.choice('use_cover_tree', ['True', 'False']),
        'dist': hp.choice('dist', ['Euclidean', 'Manhattan'])
    }
])
count = 0
best = 0

# Define minimized function
def f(params):
    global best, count
    count += 1
    acc = hyperopt_train_test(params.copy())
    if acc > best:
        print 'new best:', acc, 'using', params['type']
        best = acc
    if count % 10 == 0:
        print 'iters:', count, ', acc:', acc, 'using', params
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=30, trials=trials)
print 'best:'
print best
