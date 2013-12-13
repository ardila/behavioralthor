__author__ = 'headradio'
import cPickle
import os
import imagenet
import numpy as np
import cStringIO
feature_dir1 = '/diego_features/diego_features/a'
feature_dir2 = '/diego_features/diego_features/b'
data_dir = '/share/export/batch_storage2/batch128_img138_full'
files1 = os.listdir(feature_dir1)
files2 = os.listdir(feature_dir2)
data_files = [f for f in os.listdir(data_dir) if ~f.endswith('.meta')]
fs = imagenet.dataset.get_img_source()
extraction_name = 'challenge_set_v0'
for f1, f2, df in zip(files1, files2, data_files)[:2]:
    print f1
    print df
    fb1 = cPickle.load(open(os.path.join(feature_dir1, f1), 'rb'))['data']
    fb2 = cPickle.load(open(os.path.join(feature_dir2, f2), 'rb'))['data']
    features = np.concatenate([fb1, fb1], axis=1)
    db = cPickle.load(open(os.path.join(data_dir, df), 'rb'))
    filenames = db['filenames']
    tempfile = cStringIO.StringIO()
    for feature, filename in zip(features, filenames):
        np.save(tempfile, feature)
        name = filename+extraction_name
        print name
        fs.put(tempfile, _id=name)