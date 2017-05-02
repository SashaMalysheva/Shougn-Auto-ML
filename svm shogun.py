import os
from modshogun import GaussianKernel, GMNPSVM, LinearKernel, SigmoidKernel
from scipy.io import loadmat
from numpy import random
from modshogun import CrossValidationSplitting
import time
from modshogun import MulticlassLabels, RealFeatures
from modshogun import MulticlassAccuracy
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
                 params={'c': 1, 'kernel': 'gauss'},
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
        dictionary of 
        C: int, default = 1
            Penalty parameter of the error term
        kernel: string, default = 'gauss  
            Specifies the kernel type to be used in the algorithm.
    Nsplit: int, default = 2
            The n for n-fold cross validation.

    """
    c = params.get('c')
    if params.get('kernel' == 'gauss'):
        kernel = GaussianKernel()
        kernel.set_width(80)
    elif params.get('kernel' == 'sigmoid'):
        kernel = SigmoidKernel()
    else:
        kernel = LinearKernel()

    split = CrossValidationSplitting(labels, Nsplit)
    split.build_subsets()

    accuracy = np.zeros(Nsplit)
    time_test = np.zeros(accuracy.shape)
    for i in range(Nsplit):
        idx_train = split.generate_subset_inverse(i)
        idx_test = split.generate_subset_indices(i)

        feats.add_subset(idx_train)
        labels.add_subset(idx_train)
        svm = GMNPSVM(c, kernel, labels)
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


evaluate4svm(labels, feats).mean()
