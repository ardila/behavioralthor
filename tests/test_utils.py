__author__ = 'headradio'
import numpy as np
import tabular as tb
import dldata.metrics.utils
import behavioralthor.utils
import scipy.stats


def test_bigtrain_decision_and_margins():
    np.random.seed(0)
    nfeat = 50
    n = 3000
    F = np.random.uniform(10 * np.ones((n, nfeat)))
    y = np.ravel(np.round(np.random.uniform(2*np.ones((n, 1)))))
    meta = tb.tabarray(records=zip(y, y), names=['y', 'also_y'])

    eval_config = {
        'train_q': None,
        'test_q': None,
        'npc_train': 500,
        'npc_test': 500,
        'npc_validate': 0,
        'num_splits': 1,
        'split_by': 'y',
        'labelfunc': 'y',
        'metric_screen': 'classifier',
        'metric_kwargs': {'model_type': 'MCC2',
                          'model_kwargs': None,
                          'normalization': True,
                          'trace_normalize': False,
                          'margins': True}}
    results = dldata.metrics.utils.compute_metric_base(F, meta, eval_config, attach_models=True, return_splits=True)
    splits = results['splits']
    Xtest = F[splits[0][0]['test']]
    ytest = y[splits[0][0]['test']]
    Xtrain = F[splits[0][0]['train']]
    ytrain = y[splits[0][0]['train']]
    M1 = behavioralthor.utils.bigtrain(Xtrain, Xtest, ytrain, ytest, unnormed_margins=False)
    M2 = behavioralthor.utils.bigtrain(Xtrain, Xtest, ytrain, ytest, unnormed_margins=True)
    dldata_errors = results['split_results'][0]['test_errors']
    test_errors1 = np.sum(M1, 1) < 0
    test_errors2 = np.sum(M2, 1) < 0
    assert np.all(test_errors1 == test_errors2), \
        'The unnormed margin argument does not affect decisions'
    assert np.mean(test_errors1 == dldata_errors) > .96, \
        'Bigtrain is at least as consistent with MCC2 as MCC is with MCC2 in terms of decisions made'
    m = results['split_results'][0]['test_margins']
    tol = 1./100
    print (1-scipy.stats.pearsonr(m[0:500, 1], -M1[0:500, 1])[0])
    assert (1-scipy.stats.pearsonr(m[0:500, 1], -M1[0:500, 1])[0]) < tol, \
        'Bigtrain is at least as consistent with MCC2 as MCC is with MCC2 in terms of margins'
