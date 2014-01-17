__author__ = 'headradio'
from behavioralthor.utils import bigtrain
import imagenet
from collections import defaultdict, Counter
import cPickle
#from dldata.metrics.utils import get_subset_splits
import numpy as np
import imagenet.dldatasets as datasets
#dataset = imagenet.dldatasets.Challenge_Synsets_2_Pixel_Hard()
#dataset = imagenet.dldatasets.Big_Pixel_Screen()
#dataset = imagenet.dldatasets.HvM_Categories()


label_filename = 'not_yet'

y = cPickle.load(open('/home/ardila/synsets.p', 'rb'))
F = np.memmap('/home/ardila/features.dat', dtype='float32', mode='r', shape=(1200000, 8192))

#m = tb.tabarray([[i] for i in y], names=['synset'])
# synset_count = Counter(y)
# eval_config = {'train_q': lambda x: synset_count[x['synset']] > 550,
#                'test_q': lambda x: synset_count[x['synset']] > 550,
#                'npc_train': 500,
#                'npc_test': 50,
#                'npc_validate': 0,
#                'num_splits': 10,
#                'split_by': 'synset',
#                'labelfunc': 'synset',
#                'metric_screen': 'classifier',
#                'metric_kwargs': {'model_type': 'linear_model.SGDClassifier',
#                                  'model_kwargs': {'shuffle': True,
#                                                   'loss': "hinge",
#                                                   'alpha': 0.01,
#                                                   'n_iter': 71,
#                                                   'fit_intercept': True,
#                                                   'n_jobs': -1},
#                                  'normalization': True,
#                                  'trace_normalize': False,
#                                  'margins': False}
#                }
import tabular as tb
import dldata.metrics.utils as u
# results = u.compute_metric_base(F, m, eval_config)
#This was too slow
#splits = get_subset_splits(dataset.meta, npc_train=150, npc_tests=[50], num_splits=1, catfunc=lambda x: x['synset'])
#split = splits[0][0]

count = {}
split = defaultdict(list)
npctrain = 2
npctest = 1
for i in range(y.shape[0]):
    count[y[i]] = count.get(y[i], 0) + 1
    if i % 1000 == 0:
        print float(i)/y.shape[0]
    if count[y[i]] <= npctrain:
        split['train'].append(i)
    elif count[y[i]] <= (npctrain+npctest):
        split['test'].append(i)


screen_name = 'CHALLENGE_FEATURE'

Xtrain = np.memmap(filename='train_memmap_'+str(screen_name)+'.dat',
                   dtype='float32', mode='w+', shape=(len(split['train']), 4096*2))
Xtest = np.memmap(filename='test_memmap_'+str(screen_name)+'.dat',
                  dtype='float32', mode='w+', shape=(len(split['test']), 4096*2))
M = np.memmap(filename='margin_memmap_'+str(screen_name)+'.dat',
              dtype='float32', mode='w+', shape=(len(split['test']), len(np.unique(y))))
print 'Shape'
print len(split['test'])
print len(np.unique(y))
#Because of memory concerns, this must be done in chunks


def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

chunk_size = 200000

for chunks1, chunks2 in zip(chunks(range(Xtrain.shape[0]), chunk_size), chunks(split['train'], chunk_size)):
    print float(max(chunks1))/Xtrain.shape[0]
    Xtrain[chunks1] = F[chunks2]-np.mean(F[chunks2], axis=1)[:, np.newaxis]
for chunks1, chunks2 in zip(chunks(range(Xtest.shape[0]), chunk_size), chunks(split['test'], chunk_size)):
    print float(max(chunks1))/Xtrain.shape[0]
    Xtest[chunks1] = F[chunks2]-np.mean(F[chunks2], axis=1)[:, np.newaxis]

ytrain = y[split['train']]
ytest = y[split['test']]
M = bigtrain(Xtrain, Xtest, ytrain, ytest, unnormed_margins=True, n_jobs=-1)
filename = str(screen_name)+'margin_matrix'
np.save(filename, M)
#
#

