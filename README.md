Nowadays, Machine learning (ML) has achieved considerable successes in several applications. In order to achieve better performance it is necessary to choose a proper machine learning algorithm on a particular data set.
In this project I covered K-Nearest Neighbors (KNN), Support Vector Machines (SVM) machine learning models.
In KNN model the prediction is implemented by finding the K nearest neighbors of the query point and voting. Here K is a hyperparameter for the algorithm. Smaller values of K give the model with low bias but high variance; while larger values for K give low variance but high bias.
That is why important to tune the parameters of the model.
In contrast to KNN - SVMs attempt to model the decision function separating each class from one another. They compare examples utilising similarity measures (so-called Kernels) instead of distances like KNN does. They do not scale very well to cases with a huge number of classes but usually lead favourable results when applied to a small number of classes cases.
Therefore, I decide that it more useful to tune them all at once to get the best model overall.
