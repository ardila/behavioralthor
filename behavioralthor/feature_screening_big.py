__author__ = 'headradio'
from behavioralthor.utils import bigtrain
import imagenet
from collections import defaultdict
import cPickle
#from dldata.metrics.utils import get_subset_splits
import numpy as np


#dataset = imagenet.dldatasets.Challenge_Synsets_2_Pixel_Hard()
#dataset = imagenet.dldatasets.Big_Pixel_Screen()
#dataset = imagenet.dldatasets.HvM_Categories()


label_filename = 'not_yet'

y = cPickle.load(open('/home/ardila/synsets.p', 'rb'))
#This was too slow
#splits = get_subset_splits(dataset.meta, npc_train=150, npc_tests=[50], num_splits=1, catfunc=lambda x: x['synset'])
#split = splits[0][0]

count = {}
split = defaultdict(list)
npctrain = 400
npctest = 200
for i in range(y.shape[0]):
    count[y[i]] = count.get(y[i], 0) + 1
    if i % 1000 == 0:
        print float(i)/y.shape[0]
    if count[y[i]] <= npctrain:
        split['train'].append(i)
    elif count[y[i]] <= (npctrain+npctest):
        split['test'].append(i)


screen_name = 'CHALLENGE_FEATURE'
F = np.memmap('/home/ardila/features.dat', dtype='float32', mode='w+', shape=(1200256, 8192))[:120000]
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
    Xtrain[chunks1] = F[chunks2]
for chunks1, chunks2 in zip(chunks(range(Xtest.shape[0]), chunk_size), chunks(split['test'], chunk_size)):
    print float(max(chunks1))/Xtrain.shape[0]
    Xtest[chunks1] = F[chunks2]

ytrain = y[split['train']]
ytest = y[split['test']]
M = bigtrain(Xtrain, Xtest, ytrain, ytest, unnormed_margins=True, n_jobs=-1)
filename = str(screen_name)+'margin_matrix'
np.save(filename, M)
