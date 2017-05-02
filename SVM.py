import os
from modshogun import GaussianKernel, GMNPSVM, LinearKernel, SigmoidKernel
from scipy.io import loadmat
from numpy import random
from modshogun import CrossValidationSplitting
import time
from modshogun import MulticlassLabels, RealFeatures
from modshogun import MulticlassAccuracy
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


##############################
##==== Global variables ====##
##############################
Xtrain, Ytrain, Xtest, Ytest = get_data()
labels, feats, labels_test, feats_test = generate_labels(Xtrain, Ytrain, Xtest, Ytest)


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
        print c, kernal, labels

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
    return evaluate4svm(labels, feats, params).mean()


##############################
##==== Global variables ====##
##############################
# Define the search space.
"""
Parameters
----------
C: int in range(0, 3)
    Penalty parameter of the error term.
kernel: string - 'linear' or 'gauss' or 'sigmoid'
    Specifies the kernel type to be used in the algorithm.
"""
space4knn = {
    'c': hp.choice('c', range(0, 3)),
    'kernal': hp.choice('kernel', ['linear', 'gauss', 'sigmoid'])
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
parameters = ['c', 'kernel']
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
