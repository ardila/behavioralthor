from behavioralthor.utils import *
import imagenet
import datasets

from dldata.metrics.utils import get_subset_splits
import numpy as np


dataset = datasets.Challenge_Synsets_100_Random()
#dataset = imagenet.dldatasets.HvM_Categories()

print dataset.name


def mean_performance(margin_matrix):
    return np.sum(margin_matrix > 0) / float((margin_matrix.shape[0] * (margin_matrix.shape[1] - 1)))


def label_variance(margin_matrix, y):
    rval = []
    for i, label in enumerate(np.unique(y)):
        inds = y == label
        label_performance = mean_performance(margin_matrix[inds])
        rval.append(label_performance)
    return np.var(rval)

preprocs = [{'crop': None,
             'dtype': 'float32',
             'mask': None,
             'mode': 'RGB',
             'normalize': False,
             'resize_to': (64, 64)},
            {'crop': None,
             'dtype': 'float32',
             'mask': None,
             'mode': 'L',
             'normalize': False,
             'resize_to': (64, 64)},
            {'crop': None,
             'dtype': 'float32',
             'mask': None,
             'mode': 'RGB',
             'normalize': False,
             'resize_to': (128, 128)},
            {'crop': None,
             'dtype': 'float32',
             'mask': None,
             'mode': 'L',
             'normalize': False,
             'resize_to': (128, 128)}]

M = np.random.rand(20, 20)
ytest = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
mean_performance(M)
label_variance(M, ytest)
y = dataset.meta['synset']
# X = P[:]  # may have to memmap this
splits = get_subset_splits(dataset.meta, npc_train=150, npc_tests=[50], num_splits=1, catfunc=lambda x: x['synset'])
split = splits[0][0]

    #
    #
    # dtype='float32', mode='w+', shape=memmap_shape)
    #     return np.memmap(filename, 'float32', 'w+'))


for i, preproc in enumerate(preprocs):
    P = dataset.get_pixel_features(preproc)
    Xtrain = np.memmap(filename='train_memmap_'+str(i)+'.dat',
                       dtype='float32', mode='w+', shape=(len(split['train']), P.shape[1]))
    Xtest = np.memmap(filename='test_memmap_'+str(i)+'.dat',
                      dtype='float32', mode='w+', shape=(len(split['test']), P.shape[1]))
    Xtrain[:] = P[split['train']]
    Xtest[:] = P[split['test']]
    ytrain = y[split['train']]
    ytest = y[split['test']]
    M = bigtrain(Xtrain, Xtest, ytrain, ytest, unnormed_margins=True, n_jobs=-1)
    filename = 'preproc_unnormed_'+str(i)
    print mean_performance(M)
    print label_variance(M, ytest)
    np.save(filename, M)

np.save('preproc_screen_labels.npy', ytest)


