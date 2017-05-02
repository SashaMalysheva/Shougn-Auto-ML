import matplotlib.pyplot as plt

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Get the classic Iris data set.
iris = datasets.load_iris()
X = iris.data
y = iris.target


# Apply hyperopt library to finding the best parameters
# to a Random Forests machine learning model.
def hyperopt_train_test(params):
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X, y).mean()


##############################
##==== Global variables ====##
##############################
"""
Define the search space.

Parameters
----------
max_depth: int in range(1, 20)
    The maximum depth of the tree.
max_features: int in range(1, 5)
    The number of features to consider when looking for the best split.
n_estimators: int in range(1, 20)
    The number of trees in the forest.
criterion: string - "gini" or "entropy"
    The function to measure the quality of a split.
"""
space4rf = {
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'max_features': hp.choice('max_features', range(1, 5)),
    'n_estimators': hp.choice('n_estimators', range(1, 20)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
}
best = 0


# Define minimized function
def f(params):
    global best
    acc = hyperopt_train_test(params)
    if acc > best:
        best = acc
        print('new best:', best, params)
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f, space4rf, algo=tpe.suggest, max_evals=10, trials=trials)
print('best:')
print(best)

# Draw plot of the function.
parameters = ['n_estimators', 'max_depth', 'max_features', 'criterion']
f, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    print(i, val)
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    xs, ys = zip(*sorted(zip(xs, ys)))
    ys = np.array(ys)
    axes[int(i / 3), int(i % 3)].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i) / len(parameters)))
    axes[int(i / 3), int(i % 3)].set_title(val)
plt.show()
