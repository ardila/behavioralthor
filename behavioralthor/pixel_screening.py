from behavioralthor.utils import *
import imagenet
import datasets

from dldata.metrics.utils import get_subset_splits
import numpy as np


dataset = datasets.Challenge_Synsets_100_Random()
# dataset = imagenet.dldatasets.HvM_Categories()


def mean_performance(margin_matrix):
    np.sum(margin_matrix > 0) / (margin_matrix.shape[0] * (margin_matrix.shape[1] - 1))


def label_variance(margin_matrix, y):
    rval = []
    for i, label in enumerate(np.unique(y)):
        inds = y == label
        label_performance = mean_performance(margin_matrix[np._ix(inds, inds)])
        rval.append(label_performance)
    return np.var(rval)

preprocs = [{'crop': None,
             'dtype': 'float32',
             'mask': None,
             'mode': 'RGB',
             'normalize': True,
             'resize_to': (128, 128)},
            {'crop': None,
             'dtype': 'float32',
             'mask': None,
             'mode': 'L',
             'normalize': True,
             'resize_to': (128, 128)},
            {'crop': None,
             'dtype': 'float32',
             'mask': None,
             'mode': 'RGB',
             'normalize': True,
             'resize_to': (256, 256)},
            {'crop': None,
             'dtype': 'float32',
             'mask': None,
             'mode': 'L',
             'normalize': True,
             'resize_to': (256, 256)}]

for i, preproc in enumerate(preprocs):
    P = dataset.get_pixel_features(preproc)
    y = dataset.meta['synset']
    # X = P[:]  # may have to memmap this
    splits = get_subset_splits(dataset.meta, npc_train=150, npc_tests=[50], num_splits=1, catfunc=lambda x: x['synset'])
    for split in splits:
        split = split[0]  # don't use validation
        Xtrain = P[split['train']]
        Xtest = P[split['test']]
        ytrain = y[split['train']]
        ytest = y[split['test']]
        M = bigtrain(Xtrain, Xtest, ytrain, ytest, unnormed_margins=True, n_jobs=8)
        filename = 'preproc_'+str(i)
        print mean_performance(M)
        print label_variance(M, ytest)
        np.save(filename, M)


