__author__ = 'headradio'
import cPickle
import os
import imagenet
import numpy as np

feature_dir1 = '/diego_features/diego_features/a'
feature_dir2 = '/diego_features/diego_features/b'
fs = imagenet.dataset.get_img_source()
fp = np.memmap('features.dat', dtype='float32', mode='w+', shape=(1200000, 8192))
i = 0
for data_batch_idx in range(9375):
    fn = 'data_batch_'+str(data_batch_idx)
    print float(i)/1200000
    fb1 = cPickle.load(open(os.path.join(feature_dir1, fn), 'rb'))['data']
    fb2 = cPickle.load(open(os.path.join(feature_dir2, fn), 'rb'))['data']
    features = np.concatenate([fb1, fb1], axis=1)
#    print features
    fp[i:i+128, :] = features
    i += 128
del fp
