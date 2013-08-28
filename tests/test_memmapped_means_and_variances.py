__author__ = 'headradio'
import imagenet.dldatasets as dl
import behavioralthor.utils
import time
import os
import numpy as np
dataset = dl.HvM_Categories()


cachedir = os.getcwd()
filename = os.path.join(cachedir, 'HvM_pixel_memmap')

F = dataset.get_pixel_features()

#F = np.memmap(filename, dtype='float32', mode='w+', shape=F_larray.shape)
# F[:] = F_larray[:]
meta = dataset.meta

means, variances = behavioralthor.utils.memmapped_means_and_variances_in_parallel(F, meta, 'synset', n_jobs=8)
t = time.time()
for i in range(len(np.unique(meta['synset']))):
    idx = meta['synset'] == meta['synset'][i]
    accuracies = behavioralthor.utils.two_way_normalized_mcc_accuracy_vs_all(F[idx], i, means, variances)
    print accuracies

print time.time() - t
# print results
# print time.time() - t
# t = time.time()
# results = behavioralthor.utils.memmapped_means_and_variances_in_parallel(F_larray, meta, 'synset', n_jobs=1)
# results
#print results
# This test showed that for 8 synsets of 200 images each, extracting the means using this method is about 8 times
#  faster in parallel

