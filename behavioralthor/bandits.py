__author__ = 'headradio'
from hyperopt import base
from dldata.HvM.neural_datasets import compute_metric_base
from simffa import simffa_params
from datasets import dataset1
from pyll import scope
import numpy as np
import os
# from devthor.neural_modeling.nklike_fitting import slm_memmap
from simffa.simffa_utils import slm_memmap

@base.as_bandit()
def fruits_vs_chairs_L3(template=None):
    """

    :param template: Template of features to sample from
    :return: scope evaluation of fruits vs chair classification for dataset 1
    """
    dataset = dataset1()
    fruits_and_chairs_1 = lambda x: (x['obj'] in dataset.obj_set1) and (x['category'] in ['Fruits', 'Chairs'])
    if template is None:
        template = simffa_params.l3_params
    eval_config = {
        'npc_train': 20,
        'npc_test': 20,
        'npc_validate': 0,
        'num_splits': 5,
        'split_by': 'category',
        'catfunc': lambda x: x['category'],
        'train_q': fruits_and_chairs_1,
        'test_q': fruits_and_chairs_1,
        'metric_screen': 'classifier',
        'metric_kwargs': {'model_type': 'MCC2'}}
    return scope.dp_sym_loss(template, dataset, eval_config, feature_lambda=fruits_and_chairs_1)

@scope.define
def dp_sym_loss(template_sample, dataset, eval_config, feature_lambda=None, features=None):
    """

    :param template_sample: Sample of parameters from template
    :param dataset: Dataset object to evaluate loss for
    :param eval_config: eval_config passed to compute_metric_base (see that function for reference on this dictionary)
    :param feature_lambda: function used to determine whether to extract features for an image. Input is meta entry.
    :param features: can be used to pass existing features, features will not be rextracted
    :return: float d prime loss score
    """
    if features is None:
        features = get_features(template_sample, dataset, feature_lambda)
    results = compute_metric_base(features, dataset.meta, eval_config)
    return results['dp_sym_loss']


def get_features(template_sample, dataset, feature_lambda=None):
    """

    :param template_sample: parameters sample from a template
    :param dataset: dataset object to extract features from
    :param feature_lambda: function used to subset images. takes meta entry as input returns true or false
    :return: memmap of features
    """
    print template_sample.items()
    namebase = 'memmap_' + str(np.random.randint(100000000))
    # record['namebase'] = namebase
    Images = dataset.get_images()
    if feature_lambda is not None:
        inds = [feature_lambda(entry) for entry in dataset.meta]
        Images = Images[inds]
    if 'BASEDIR' in os.environ:
        basedir = os.environ['BASEDIR']
    else:
        basedir = None
    # record['basedir'] = basedir
    print "TEST"
    features = slm_memmap(
        desc=template_sample['desc'],
        X=Images,
        basedir=basedir,
        name=namebase + '_img_feat')
    # fs = features.shape
    # num_features = np.prod(fs[1:])
    # record['feature_shape'] = fs
    # record['num_features'] = num_features
    features = features[:]
    #features = features.reshape((fs[0], num_features))
    return features


