import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Get the classic Iris data set.
iris = datasets.load_iris()
X = np.array(iris.data)
y = np.array(iris.target)
print(X.shape)

# Apply hyperopt library to finding the best parameters
# to a K-Nearest Neighbor (KNN) machine learning model.
def hyperopt_train_test(params):
    return cross_val_score(KNeighborsClassifier(**params), X, y).mean()


"""
Define the search space.

Parameters
----------
n_neighbors: int in range(1, 20)
    Number of neighbors.
"""
space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1, 20))
}


# Define minimized function
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
print('best:')
print(best)

