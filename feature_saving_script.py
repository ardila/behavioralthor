__author__ = 'headradio'
import cPickle
import os
import imagenet
import numpy as np
import cStringIO
feature_dir1 = '/diego_features/diego_features/a'
feature_dir2 = '/diego_features/diego_features/b'
files1 = [f for f in os.listdir(feature_dir1) if f.startswith('data_batch')]
files2 = [f for f in os.listdir(feature_dir2) if f.startswith('data_batch')]
fs = imagenet.dataset.get_img_source()
extraction_name = 'challenge_set_v0'
fp = np.memmap('features.dat', dtype='float32', mode ='w+', shape=(1200256, 8192))
i = 1
for f1, f2 in zip(files1, files2):
    print f1
    print float(i)/1200256
    fb1 = cPickle.load(open(os.path.join(feature_dir1, f1), 'rb'))['data']
    fb2 = cPickle.load(open(os.path.join(feature_dir2, f2), 'rb'))['data']
    features = np.concatenate([fb1, fb1], axis=1)
    fp[i:i+128, :] = features
    i += 128
    # db = cPickle.load(open(os.path.join(data_dir, df), 'rb'))
    # filenames = db['filenames']
    # tempfile = cStringIO.StringIO()
    # for feature, filename in zip(features, filenames):
    #     np.save(tempfile, feature)
    #     name = filename+extraction_name
    #     if not fs.exists(name):
    #         fs.put(tempfile, _id=name)