__author__ = 'headradio'
from hyperopt import base
from dldata.HvM.neural_datasets import compute_metric_base
from devthor import devthor_new_new_params
from datasets import dataset1
from pyll import scope
import numpy as np
import os
from devthor.neural_modeling.nklike_fitting import slm_memmap
# from simffa.simffa_utils import slm_memmap

Broken_objects = [
    'jewelry_41',
    'Workman_pose10',
    'Workman_pose01',
    'Soldier_pose08',
    'Soldier_pose02',
    'SWAT_pose06',
    'Professor_pose04',
    'Policewoman_pose02',
    'Nurse_posing_T',
    'Medic_pose11',
    'Fireman_pose08',
    'Fireman_pose01',
    'Engineer_pose08',
    'Engineer_pose01',
    'Air_hostess_pose02',
    'Air_hostess_pose01',
    '04_piano',
    '02_viola',
    '29_violin_modern',
    '054M',
    '093M',
    '082M',
    'MB31044',
    'MB31036',
    'MB30019',
    'MB29949',
    'MB28889',
    'MB30810',
    'MB31131',
    'MB29870',
    'MB28886',
    'MB27471',
    'MB28969',
    'MB30660',  # New from here
    'jewelry_08',
    '58_drums',
    'jewelry_22',
    '041M',
    'jewelry_35',
    '17_el_guitar',
    'jewelry_26b']


@base.as_bandit()
def fruits_vs_chairs_L3(template=None):
    """

    :param template: Template of features to sample from
    :return: scope evaluation of fruits vs chair classification for dataset 1
    """
    dataset = dataset1()
    fruits_and_chairs_1 = lambda x: (x['category'] in frozenset(['fruits', 'chairs'])) and \
                                    (x['obj'] in frozenset(dataset.obj_set1)) and \
                                    (x['obj'] not in frozenset(Broken_objects))
    # fruits_and_chairs_1 = {'obj': dataset.obj_set1, 'category': ['Fruits', 'Chairs']}
    if template is None:
        template = devthor_new_new_params.l3_params
    eval_config = {
        'npc_train': 15,
        'npc_test': 5,
        'npc_validate': 0,
        'num_splits': 4,
        'split_by': 'category',
        'labelfunc': 'category',
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
    record = {'loss': results['dp_sym_loss'], 'results': results, 'status': 'ok'}
    return record


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
    fs = features.shape
    features = features.reshape((fs[0], np.prod(fs[1:])))
    #features = features.reshape((fs[0], num_features))
    return features
