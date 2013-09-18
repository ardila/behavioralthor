__author__ = 'headradio'
from hyperopt import base
from dldata.metrics.utils import compute_metric_base
from devthor import devthor_new_new_params
from datasets import dataset1
from pyll import scope
import numpy as np
import os
from devthor.neural_modeling.nklike_fitting import slm_memmap, slm_bandit_exceptions
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


@base.as_bandit(exceptions=slm_bandit_exceptions)
def fruits_vs_chairs_L1R(template=None):
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
        template = devthor_new_new_params.l1r_params
    eval_config = {
        'npc_train': 40,
        'npc_test': 40,
        'npc_validate': 0,
        'num_splits': 4,
        'split_by': 'category',
        'labelfunc': 'category',
        'train_q': fruits_and_chairs_1,
        'test_q': fruits_and_chairs_1,
        'metric_screen': 'classifier',
        'metric_kwargs': {'model_type': 'MCC2'}}
    return scope.dp_sym_loss(template, dataset, eval_config)

@base.as_bandit(exceptions=slm_bandit_exceptions)
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
        'npc_train': 40,
        'npc_test': 40,
        'npc_validate': 0,
        'num_splits': 4,
        'split_by': 'category',
        'labelfunc': 'category',
        'train_q': fruits_and_chairs_1,
        'test_q': fruits_and_chairs_1,
        'metric_screen': 'classifier',
        'metric_kwargs': {'model_type': 'MCC2'}}
    return scope.dp_sym_loss(template, dataset, eval_config)


@base.as_bandit(exceptions=slm_bandit_exceptions)
def object_level_fruits_and_chairs_L3(template=None):
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
        'split_by': 'obj',
        'labelfunc': 'obj',
        'train_q': fruits_and_chairs_1,
        'test_q': fruits_and_chairs_1,
        'metric_screen': 'classifier',
        'metric_kwargs': {'model_type': 'MCC2'}}
    return scope.dp_sym_loss(template, dataset, eval_config)


@base.as_bandit(exceptions=slm_bandit_exceptions)
def fruits_vs_chairs_L2(template=None):
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
        template = devthor_new_new_params.l2_params
    eval_config = {
        'npc_train': 40,
        'npc_test': 40,
        'npc_validate': 0,
        'num_splits': 4,
        'split_by': 'category',
        'labelfunc': 'category',
        'train_q': fruits_and_chairs_1,
        'test_q': fruits_and_chairs_1,
        'metric_screen': 'classifier',
        'metric_kwargs': {'model_type': 'MCC2'}}
    return scope.dp_sym_loss(template, dataset, eval_config)


@base.as_bandit(exceptions=slm_bandit_exceptions)
def eight_way_imagenet_screen(template=None):
    dataset = Imagenet
    splits, labels = dataset.get_eight_way_splits()
    if template is None:
        template = devthor_new_new_params.l4_params
    eval_config = {
        'precomp_splits': splits,
        'validations': [],
        'labels': labels,
        'metric_screen': 'classifier',
        'metric_kwargs': {'model_type': 'MCC2'}
    }
    return scope.dp_sym_loss(template, dataset, eval_config)

@base.as_bandit(exceptions=slm_bandit_exceptions)
def fruits_vs_chairs_L4(template=None):
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
        template = devthor_new_new_params.l4_params
    eval_config = {
        'npc_train': 40,
        'npc_test': 40,
        'npc_validate': 0,
        'num_splits': 4,
        'split_by': 'category',
        'labelfunc': 'category',
        'train_q': fruits_and_chairs_1,
        'test_q': fruits_and_chairs_1,
        'metric_screen': 'classifier',
        'metric_kwargs': {'model_type': 'MCC2'}}
    return scope.dp_sym_loss(template, dataset, eval_config)


@scope.define
def dp_sym_loss(template_sample, dataset, eval_config, features=None, delete_memmaps=True):
    """

    :param template_sample: Sample of parameters from template
    :param dataset: Dataset object to evaluate loss for
    :param eval_config: eval_config passed to compute_metric_base (see that function for reference on this dictionary)
    :param features: can be used to pass existing features, features will not be rextracted
    :return: float d prime loss score
    """
    record = {}
    if features is None:
        features, record = get_features(template_sample, dataset, record)
    #Code to flatten features for computing metrics
    features, meta = subset_and_reshape(features, dataset.meta, eval_config)
    results = compute_metric_base(features, meta, eval_config)
    if delete_memmaps:
        if record['basedir'] is not None:
            path = os.path.join(record['basedir'],
                                record['namebase'] + '_img_feat')
        else:
            path = record['namebase'] + '_img_feat'
        print ('removing %s' % path)
        os.system('rm -rf %s' % path)
    record['loss'] = results['dp_sym_loss']
    record['results'] = results
    record['status'] = 'ok'
    record['desc'] = template_sample
    return record


def get_features(template_sample, dataset, record=None):
    """
    :param template_sample: parameters sample from a template
    :param dataset: dataset object to extract features from
    :return: memmap of features
    """
    if record is None:
        record = {}
    print template_sample
    namebase = 'memmap_' + str(hash(repr(template_sample)))
    record['namebase'] = namebase
    Images = dataset.get_images()
    if 'BASEDIR' in os.environ:
        basedir = os.environ['BASEDIR']
    else:
        basedir = None
        record['basedir'] = basedir
    print "TEST"
    features = slm_memmap(
        desc=template_sample['desc'],
        X=Images,
        basedir=basedir,
        name=namebase + '_img_feat')
    return features, record


def subset_and_reshape(features, meta, eval_config):
    if eval_config['train_q'] == eval_config['test_q']:
            inds = filter(lambda ind: eval_config['train_q'](meta[ind]), np.arange(len(meta)))
            eval_config['train_q'] = lambda x: True
            eval_config['test_q'] = lambda x: True
    else:
        raise NotImplementedError
    features = features[inds]
    meta = meta[inds]
    fs = features.shape
    num_features = np.prod(fs[1:])
    features = features.reshape((fs[0], num_features))
    return features, meta
