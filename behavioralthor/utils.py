"""
Various routines that are helpful for screening models/datasets
"""
import numpy as np
import itertools
from joblib import Parallel, delayed
from dldata.metrics.utils import compute_metric
from multiprocessing import Pool
import functools
import os
import cPickle
from scipy.stats import norm


# def two_normed_correlation(F, meta, means, vars):
def bigtrain(Xtrain, Xtest, ytrain, ytest, unnormed_margins=False, means_and_variances=None, n_jobs=1):
    """
    A reference implementation for how you can fairly efficiently run many
    binary MCC problems simultaneously.

    :param Xtrain: training features, array, shape = (ntrain , nfeat)
    :param Xtest: testing features, array, shape = (ntest, nfeat)
    :param ytrain: training labels, array, shape = (ntrain,)
    :param ytest: testing labels, array, shape = (ntest, )
    :param means_and_variances:
    :param n_jobs: how many parallel workers to use.

    Returns:
    P -- the "margin matrix", array, shape = (ntest, ncat), where the value
               sign(P[i, j])
         is 1 if test image i is predicted correctly in the binary problem
         comparing the category that test image i actually is, versus category j.

    """

    ###NB: this function assumes the training split is balanced.

    cats = np.unique(ytrain)                # get the categories
    nfeat = Xtrain.shape[1]                 # get the number of features
    N = len(cats)                           # get the number of categories

    M = np.empty((N, nfeat))                  # preallocate mean matrix
    V = np.empty((N, nfeat))                  # preallocate variance matrix
    P = np.empty((Xtest.shape[0], N))         # preallocate margin matrix
    print 'calculating means and variances'
    # "Training" i.e. get means and variances for every class
    if means_and_variances is None:
        means_and_variances = memmapped_means_and_variances_in_parallel(Xtrain, ytrain, cats, n_jobs)
    M, V = means_and_variances
    #"testing"
    print 'testing'
    #again, this could be parallelized in various ways
    tind = 0

    results = Parallel(n_jobs=n_jobs, verbose=100)(delayed(get_class_margins)(Xtest[ytest == k], kind, M, V, unnormed_margins)
                                      for kind, k in enumerate(cats))
    for kind, k in enumerate(cats):
        class_idx = ytest == k
        P[class_idx] = results[kind]
    return P
    # for kind, k in enumerate(cats):
    #     #get the feature-wise means for all the binary problems at once
    #     m1 = 0.5 * (M[kind] + M)
    #
    #     #get the feature-wise stds for all the binary problems at once
    #     s1 = np.sqrt(0.5 * (V[kind]) + 0.5 * V +
    #                  0.25 * (M[kind]**2) - 0.5 * (M[kind] * M) + 0.25 * (M**2))
    #
    #     #get the per-category normalized means, doing all normalizations at once
    #     m0 = 0.5 * (M[kind] - M) / s1
    #     #normalize the means to be of norm 1
    #     m0 = m0/np.apply_along_axis(np.linalg.norm, 1, m0)[:, np.newaxis]
    #     #now, for each test image of category k, do:
    #     class_idx = ytest == k
    #     Xk = Xtest[class_idx]
    #     P[class_idx] = [normalize_and_dot(x, m1, s1, m0, unnormed_margins) for x in Xk]
    #     # P[class_idx] = Parallel(n_jobs=n_jobs)(delayed(normalize_and_dot)
    #     #                                       (x, m1, s1, m0, unnormed_margins) for x in Xk)
    #     P[class_idx, kind] = 0
    # return P


def get_class_margins(Xk, kind, M, V, unnormed_margins):
    #get the feature-wise means for all the binary problems at once
    m1 = 0.5 * (M[kind] + M)

    #get the feature-wise stds for all the binary problems at once
    s1 = np.sqrt(0.5 * (V[kind]) + 0.5 * V +
                 0.25 * (M[kind]**2) - 0.5 * (M[kind] * M) + 0.25 * (M**2))

    #get the per-category normalized means, doing all normalizations at once
    m0 = 0.5 * (M[kind] - M) / s1
    #normalize the means to be of norm 1
    m0 = m0/np.apply_along_axis(np.linalg.norm, 1, m0)[:, np.newaxis]
    #now, for each test image of category k, do:
    P = np.asarray([normalize_and_dot(x, m1, s1, m0, unnormed_margins) for x in Xk])
    # P[class_idx] = Parallel(n_jobs=n_jobs)(delayed(normalize_and_dot)
    #                                       (x, m1, s1, m0, unnormed_margins) for x in Xk)
    P[:, kind] = 0
    return P


def normalize_and_dot(x, m1, s1, m0, unnormed_margins):
    xn = (x-m1)/s1
    if not unnormed_margins:
        xn = xn/np.apply_along_axis(np.linalg.norm, 1, xn)[:, np.newaxis]
    return np.sum(xn*m0, 1)


def elementwise_distances(A, B):
    """
    returns correlation distances on a vector-by-vector basis

    Arguments:
        A -- array (or list) of vectors
        B -- array (or list) of vectors of same shape as A

    Returns:
        array of scalars of same length as A and B, whose i-th value is the
        correlation between A[i] and B[i]

    """
    ##this function can and should be optimized!!!
    pf = lambda x: np.corrcoef(*x)[0, 1]
    return np.array(map(pf, zip(A, B)))


def test():
    #situation with 4-dimensional features and 3 classes (labeled 0, 1, 2)
    K = 4

    #allocating 600 training vectors
    #200 vectors each for the three classes
    # with 1's in the column associated with the label number
    N = 200
    Xtrain = np.zeros((3*N, K))
    Xtrain[:N, 0] = 1
    Xtrain[N:2*N, 1] = 1
    Xtrain[2*N:, 2] = 1
    Xtrain += 0.01 * np.random.RandomState(0).uniform(size=(3*N, K))
    ytrain = np.concatenate([np.zeros(N), np.ones(N), 2 * np.ones(N)])

    #allocating 30 testing vectors
    N1 = 30
    Xtest = np.zeros((3*N1, K))
    Xtest[:N1/2, 0] = 1               #the first 5 vectors are "correct"
    Xtest[N1/2:N1, 1] = 1             #the second 5 vectors are "wrong" -- they'll be confused with category 1
    Xtest[N1:2*N1, 1] = 1             #the 10-20th vectors are all "correct"
    Xtest[2*N1:2*N1 + N1 / 2, 2] = 1  #the 20-25th vectors are "correct"
    Xtest[2*N1 + N1 / 2:, 1] = 1      #the 25-30th vectors are "wrong" -- they'll be confused with category 1
    Xtest += 0.01 * np.random.RandomState(1).uniform(size=(3*N1, K))
    ytest = np.concatenate([np.zeros(N1), np.ones(N1), 2 * np.ones(N1)])

    #get the margins
    P = bigtrain(Xtrain, Xtest, ytrain, ytest, unnormed_margins=True, n_jobs=1)

    #note that P[0:5, 1] is positive, but P[5:10, 1] is negative
    #note that P[10:20, [0, 2]] are all positive
    #note that P[20:25, 1] is positive but P[25:30, 1] is negative
    return P


def two_way_normalized_mcc_accuracy_vs_all(class_features, i, means, variances):
    n_classes = means.shape[0]
    accuracies = np.zeros(n_classes)
    for j in range(n_classes):
        if i != j:
            unnormed_margins = np.dot(class_features,  (means[i]-means[j])/(variances[i]+variances[j]))
            decisions = np.sign(unnormed_margins)
            accuracy = np.mean(decisions < 0)
            accuracies[j] = accuracy
    return accuracies


def memmapped_means_and_variances_in_parallel(X, y, cats, n_jobs=1):
    cachedir = os.getcwd()
    n_features = X.shape[1]
    n_classes = len(cats)
    memmap_shape = (n_classes, n_features)
    mean_memmap = np.memmap(os.path.join(cachedir, 'means'),
                            dtype='float32', mode='w+', shape=memmap_shape)
    var_memmap = np.memmap(os.path.join(cachedir, 'vars'),
                           dtype='float32', mode='w+', shape=memmap_shape)
    results = Parallel(n_jobs=n_jobs)(delayed(save_mean_and_var)
                                     (X[y == label], label, cachedir)
                                      for label in cats)

    for index in range(len(results)):
        mean_memmap[index] = np.load(results[index][0])
        var_memmap[index] = np.load(results[index][1])
    return mean_memmap, var_memmap
    # map(lambda x: load_to_memmap(x, mean_memmap), enumerate(mean_filenames))
    # map(lambda x: load_to_memmap(x, var_memmap), enumerate(var_filenames))
    # return mean_memmap, var_memmap


def save_mean_and_var(class_features, label, path):
    """

    :param class_features: Features from one class
    :return: mean feature vector and variance of each feature in a vector
    """
    label = str(label)
    print label + ' is loaded!'
    mean = np.mean(class_features, 0)
    var = np.var(class_features, 0)
    var = np.maximum(var, 1e-8)  # might not be neccesary, but better to play it safe
    mean_filename = os.path.join(path, label+'_mean.npy')
    var_filename = os.path.join(path, label+'_var.npy')
    np.save(mean_filename, mean)
    np.save(var_filename, var)
    return mean_filename, var_filename


def distance_matrix_from_margin_matrix(M):
    """
    :param M: #images by #categories
        Margin matrix which stores margin relative to correct decision (positive is correct)
        for a 2 way balanced task
    :return: array, #category by # category.
        CM[i,j] = proportion of stimuli from category i confused with category j
    """
    assert np.sum(M == 0) == M.shape[0], 'There are multiple 0 margins for 1 or more images'
    # Decisions are not included for class vs itself
    correct_decision = M >= 0
    y = np.array([int(np.argwhere(margin == 0)) for margin in M])
    #Distance between two synsets is the average accuracy in a 2 way balanced task
    rates = np.array([np.mean(correct_decision[y == i], 0) for i in np.unique(y)])
    # rates = (rates+rates.T) / 2
    dprimes = norm.ppf(rates) - norm.ppf(1-rates.T)
    np.fill_diagonal(dprimes, 0)
    return dprimes


def densest_cluster(distance_matrix, cluster_size, hierarchical=True):
    indexes = []
    if hierarchical:
        centroid = np.argmin(np.sum(distance_matrix, 0))
        indexes.append(centroid)
        in_cluster = range(distance_matrix.shape[0]) == centroid
        for index_of_point_to_add in range(cluster_size-1):
            if index_of_point_to_add % 100 == 0:
                print float(index_of_point_to_add/cluster_size)
            distances_to_cluster = np.mean(distance_matrix[np.ix_(in_cluster, ~in_cluster)], 0)
            closest_point = np.argmin(distances_to_cluster)
            index = np.argwhere(~in_cluster)[closest_point]
            in_cluster[index] = True
            indexes.append(index)
    else:
        means_and_inds = [(np.mean(distance_matrix[ind]), ind) for ind in range(distance_matrix.shape[0])]
        sorted_inds = [means_and_ind[1] for means_and_ind in means_and_inds]
        cluster_inds = set(sorted_inds[:cluster_size-1])
        in_cluster = [i in cluster_inds for i in range(distance_matrix.shape[0])]

    return indexes


def sorted_distance_matrix_from_margin_matrix(M):
    D = distance_matrix_from_margin_matrix(M)
    new_order = np.argsort(np.mean(D, 1))
    D = D[new_order]
    D = D[:, new_order]
    return D


def density_curve(D):
    indexes = densest_cluster(D, D.shape[0])
    return [np.mean(D[np.ix_(cluster, cluster)]) for cluster in [indexes[:i] for i in range(D.shape[0])]]


