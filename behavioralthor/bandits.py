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
        template = simffa_params.l3_params
    splits = dataset.fruit_vs_chair_splits
    eval_config = {'precomp_splits': splits, 'metric_screen': 'classifier', 'metric_kwargs': {'model_type': 'MCC2'}}
    return scope.dp_sym_loss(template, images, dataset.meta, eval_config)