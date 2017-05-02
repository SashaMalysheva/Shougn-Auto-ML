# coding=utf-8
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Get the classic Iris data set.
digits = datasets.load_digits()
X = digits.data
y = digits.target



def hyperopt_train_test(params):
    # Automatically tune the parameters of models - SVM and KNN.
    t = params['type']
    del params['type']
    if t == 'svm':
        clf = SVC(**params)
    elif t == 'knn':
        clf = KNeighborsClassifier(**params)
    else:
        return 0
    return cross_val_score(clf, X, y).mean()

"""
Define the search space.

Parameters
----------
type == 'svm'
    c: float in range(0, 10)
        Penalty parameter C of the error term.
    kernel: string - 'linear' or 'rbf'
        Specifies the kernel type to be used in the algorithm.
    gamma: float in range(0, 20)
        Kernel coefficient for ‘rbf’.
type == 'knn'
    n_neighbors: int in range(1, 20)
        Number of neighbors.
"""
space = hp.choice('classifier_type', [
    {
        'type': 'svm',
        'C': hp.uniform('C', 0, 10.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20.0)
    },
    {
        'type': 'knn',
        'n_neighbors': hp.choice('knn_n_neighbors', range(1,50))
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
    if count % 50 == 0:
        print 'iters:', count, ', acc:', acc, 'using', params
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=100, trials=trials)
print 'best:'
print best