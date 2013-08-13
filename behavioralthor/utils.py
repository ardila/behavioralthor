"""
Various routines that are helpful for screening models
"""
import numpy as np
import itertools
from joblib import Parallel, delayed
from dldata.metrics.utils import compute_metric


def compute_all_2_ways_in_field(F, dataset, field='synset', n_jobs=1):
    meta = dataset.meta
    synsets = np.unique(meta[field])
    two_ways = itertools.combinations(synsets, 2)
    eval_configs = []
    for two_way in two_ways:
        #TODO: Make this extendable from a base_eval_config
        print two_way
        eval_configs.append({
            'train_q': {field: [two_way[0], two_way[1]]},
            'test_q': {field: [two_way[0], two_way[1]]},
            'npc_train': 5,   # We need to set these numbers
            'npc_test': 1,    #
            'num_splits': 2,  #
            'npc_validate': 0,
            'split_by': field,
            'labelfunc': lambda x: (x[field], None),
            'metric_screen': 'classifier',
            'metric_kwargs': {'model_type': 'MCC2', 'normalization': True}
        })

    print dataset.meta.dtype
    print compute_metric(F, dataset, eval_configs[0])
    return Parallel(n_jobs=n_jobs)(delayed(compute_metric)
                                          (F, dataset, eval_config) for eval_config in eval_configs)