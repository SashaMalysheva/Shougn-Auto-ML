import os
import numpy as np
from scipy.io import loadmat
from numpy import random
from modshogun import MulticlassLabels, RealFeatures
from modshogun import KNN, EuclideanDistance
from modshogun import KNN_COVER_TREE, KNN_BRUTE
SHOGUN_DATA_DIR = os.getenv('SHOGUN_DATA_DIR', '/home/sasha/workspace/shogun/data')



def get_data():
    """
    Get a random subset of 1000 samples from the USPS digit recognition dataset.
    """
    mat = loadmat(os.path.join(SHOGUN_DATA_DIR, 'multiclass/usps.mat'))
    Xall = mat['data']
    Yall = np.array(mat['label'].squeeze(), dtype=np.double)

    # map from 1..10 to 0..9, since shogun
    # requires multiclass labels to be 0, 1, ..., K-1
    Yall = Yall - 1
    random.seed(0)
    subset = random.permutation(len(Yall))

    Xtest = Xall[:, subset[5000:6000]]
    Ytest = Yall[subset[5000:6000]]

    labels = MulticlassLabels(Ytest)
    feats = RealFeatures(Xtest)
    return labels, feats


def evaluate4knn(labels,
             feats,
             use_cover_tree=False,
             Nsplit=2,
             all_ks=range(1, 21)):
    """
        Run Cross-validation to the evaluation the KNN.

        Parameters
        ----------
        labels: 2d array
            Data set labels.
        feats: array
            Data set feats.
        use_cover_tree: boolean, default = False
            The flag of enable or disable this Cover Trees.
        Nsplit: int, default = 2
            Tne n for n-fold cross validation.
        all_ks: range of int, default = range(1, 21)
            Numbers of neighbors.
    """
    from modshogun import MulticlassAccuracy, CrossValidationSplitting
    import time
    split = CrossValidationSplitting(labels, Nsplit)
    split.build_subsets()

    accuracy = np.zeros((Nsplit, len(all_ks)))
    acc_train = np.zeros(accuracy.shape)
    time_test = np.zeros(accuracy.shape)
    for i in range(Nsplit):
        idx_train = split.generate_subset_inverse(i)
        idx_test = split.generate_subset_indices(i)

        for j, k in enumerate(all_ks):
            # print "Round %d for k=%d..." % (i, k)
            feats.add_subset(idx_train)
            labels.add_subset(idx_train)

            dist = EuclideanDistance(feats, feats)
            knn = KNN(k, dist, labels)
            knn.set_store_model_features(True)
            if use_cover_tree:
                knn.set_knn_solver_type(KNN_COVER_TREE)
            else:
                knn.set_knn_solver_type(KNN_BRUTE)
            knn.train()

            evaluator = MulticlassAccuracy()
            pred = knn.apply_multiclass()
            acc_train[i, j] = evaluator.evaluate(pred, labels)

            feats.remove_subset()
            labels.remove_subset()
            feats.add_subset(idx_test)
            labels.add_subset(idx_test)

            t_start = time.clock()
            pred = knn.apply_multiclass(feats)
            time_test[i, j] = (time.clock() - t_start) / labels.get_num_labels()

            accuracy[i, j] = evaluator.evaluate(pred, labels)

            feats.remove_subset()
            labels.remove_subset()
    return {'eout': accuracy, 'ein': acc_train, 'time': time_test}


labels, feats = get_data()
wo_ct = evaluate4knn(labels, feats)
