__author__ = 'headradio'
from hyperopt import base
from dldata.HvM.neural_datasets import compute_metric_base
from simffa import simffa_params
from simffa.simffa_utils import get_features
from datasets import dataset1
from pyll import scope


@scope.define
def dp_sym_loss(config, images, meta, eval_config):
    features = get_features(images, config, verbose=False)
    results = compute_metric_base(features, meta, eval_config)
    return results['dp_sym_loss']


@base.as_bandit()
def fruits_vs_chairs_L3(template=None):
    dataset = dataset1()
    images = dataset.get_images()
    if template is None:
        template = simffa_params.l3_pamporams
    eval_config = {
        'npc_train': 20,
        'npc_test': 20,
        'num_splits': 5,
        'catfunc': lambda x: x['category'],
        'train_q': lambda x: (x['obj'] in dataset.obj_set1) and (x['category'] in ['Fruits', 'Chairs']),
        'test_q': [lambda x: (x['obj'] in dataset.obj_set1) and (x['category'] in ['Fruits', 'Chairs'])],
        'metric_screen': 'classifier',
        'metric_kwargs': {'model_type': 'MCC2'}}
    return scope.dp_sym_loss(template, images, dataset.meta, eval_config)