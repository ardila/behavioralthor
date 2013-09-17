__author__ = 'headradio'
from behavioralthor.utils import *
import imagenet
from collections import defaultdict
from dldata.metrics.utils import get_subset_splits
import numpy as np


#dataset = imagenet.dldatasets.Challenge_Synsets_2_Pixel_Hard()
dataset = imagenet.dldatasets.Big_Pixel_Screen()
#dataset = imagenet.dldatasets.HvM_Categories()

preproc = {'crop': None,
           'dtype': 'float32',
           'mask': None,
           'mode': 'RGB',
           'normalize': True,
           'resize_to': (64, 64)}

y = dataset.meta['synset']
#This was too slow
#splits = get_subset_splits(dataset.meta, npc_train=150, npc_tests=[50], num_splits=1, catfunc=lambda x: x['synset'])
#split = splits[0][0]

count = {}
split = defaultdict(list)
npctrain = 150
npctest = 50
for i in range(y.shape[0]):
    count[y[i]] = count.get(y[i], 0) + 1
    if i % y.shape[0] == 100:
        print float(i)/y.shape[0]
    if count[y[i]] <= npctrain:
        split['train'].append(i)
    elif count[y[i]] <= (npctrain+npctest):
        split['test'].append(i)
#quick tests of this adhoc get_subset_splits
assert len(split['test']) == 50*len(dataset.synset_list)
assert len(split['train']) + len(split['test']) == np.unique(y).shape[0]*(npctrain + npctest)

screen_name = 'BIG_PIXEL'
P = dataset.get_pixel_features(preproc)
Xtrain = np.memmap(filename='train_memmap_'+str(screen_name)+'.dat',
                   dtype='float32', mode='w+', shape=(len(split['train']), P.shape[1]))
Xtest = np.memmap(filename='test_memmap_'+str(screen_name)+'.dat',
                  dtype='float32', mode='w+', shape=(len(split['test']), P.shape[1]))
M = np.memmap(filename='margin_memmap_'+str(screen_name)+'.dat',
              dtype='float32', mode='w+', shape=(len(split['test']), len(np.unique(y))))
print 'Shape'
print len(split['test'])
print len(np.unique(y))
Xtrain[:] = P[split['train']]
Xtest[:] = P[split['test']]
ytrain = y[split['train']]
ytest = y[split['test']]
M = bigtrain(Xtrain, Xtest, ytrain, ytest, unnormed_margins=True, n_jobs=-1)
filename = str(screen_name)+'margin_matrix'
np.save(filename, M)
