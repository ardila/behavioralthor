# %load_ext autoreload
# %autoreload 2
import imagenet
import behavioralthor
import numpy as np
import cPickle


dataset = imagenet.dldatasets.Challenge_Synsets_20_Pixel_Hard()


preproc = {'normalize': False,
           'dtype': 'float32',
           'mask': None, 'crop': None, 'flatten': True, 'resize_to': (64, 64), 'mode': 'RGB'}
lpF = dataset.get_pixel_features(preproc)
# #pF = memmap('pixel_features.dat', 'float32', 'w+', shape=lpF.shape)
# pF = lpF[:]
#
# npc_train_list = np.round(np.linspace(5, 700, 20))
# eval_config_base = {'train_q': None,
#                     'test_q': None,
#                     'npc_test': 20,
#                     'npc_validate': 0,
#                     'num_splits': 10,
#                     'split_by': 'synset',
#                     'labelfunc': 'synset',
#                     'metric_screen': 'classifier',
#                     'metric_kwargs': {'model_type': 'MCC2',
#                                       'model_kwargs': None,
#                                       'normalization': False,
#                                       'trace_normalize': True,
#                                       'margins': False}
#                     }
# pixel_results = behavioralthor.utils.training_curve(pF, dataset, npc_train_list, eval_config_base)
# cPickle.dump(pixel_results, open('pixel_results.p', 'wb'))
# pF = None  # clearing for memory's sake
# F = dataset.get_hmo_feats0()
# results = behavioralthor.utils.training_curve(F, dataset, npc_train_list, eval_config_base)
# cPickle.dump(results, open('results.p', 'wb'))
#
# pF = lpF[:]
# npc_train_list = np.round(np.linspace(5, 700, 20))
# eval_config_base = {'train_q': None,
#                     'test_q': None,
#                     'npc_test': 20,
#                     'npc_validate': 0,
#                     'num_splits': 10,
#                     'split_by': 'synset',
#                     'labelfunc': 'synset',
#                     'metric_screen': 'classifier',
#                     'metric_kwargs': {'model_type': 'svm.LinearSVC',
#                                       'model_kwargs': {'C': .0005},
#                                       'normalization': False,
#                                       'trace_normalize': True,
#                                       'margins': False}
#                     }
# pixel_results = behavioralthor.utils.training_curve(pF, dataset, npc_train_list, eval_config_base)
# cPickle.dump(pixel_results, open('svm_pixel_results_svm.p', 'wb'))
# pF = None
# F = dataset.get_hmo_feats0()
# results = behavioralthor.utils.training_curve(F, dataset, npc_train_list, eval_config_base)
# cPickle.dump(results, open('svm_results_svm.p', 'wb'))


pF = lpF[:]
npc_train_list = np.round(np.linspace(5, 700, 20))
eval_config_base = {'train_q': None,
                    'test_q': None,
                    'npc_test': 20,
                    'npc_validate': 0,
                    'num_splits': 10,
                    'split_by': 'synset',
                    'labelfunc': 'synset',
                    'metric_screen': 'classifier',
                    'metric_kwargs': {'model_type': 'neighbors.KNeighborsClassifier',
                                      'model_kwargs': None,
                                      'normalization': False,
                                      'trace_normalize': True,
                                      'margins': False}
                    }
pixel_results = behavioralthor.utils.training_curve(pF, dataset, npc_train_list, eval_config_base)
cPickle.dump(pixel_results, open('nn_pixel_results.p', 'wb'))
pF = None
F = dataset.get_hmo_feats0()
results = behavioralthor.utils.training_curve(F, dataset, npc_train_list, eval_config_base)
cPickle.dump(results, open('nn_results.p', 'wb'))


pF = lpF[:]
npc_train_list = np.round(np.linspace(5, 700, 20))
eval_config_base = {'train_q': None,
                    'test_q': None,
                    'npc_test': 20,
                    'npc_validate': 0,
                    'num_splits': 10,
                    'split_by': 'synset',
                    'labelfunc': 'synset',
                    'metric_screen': 'classifier',
                    'metric_kwargs': {'model_type': 'linear_model.SGDClassifier',
                                      'model_kwargs': {'shuffle': True,
                                                       'loss': "hinge",
                                                       'alpha': 0.01,
                                                       'n_iter': 71,
                                                       'fit_intercept': True,
                                                       'n_jobs': -1},
                                      'normalization': True,
                                      'trace_normalize': False,
                                      'margins': False}
                    }
pixel_results = behavioralthor.utils.training_curve(pF, dataset, npc_train_list, eval_config_base)
cPickle.dump(pixel_results, open('sgd_pixel_results.p', 'wb'))
pF = None
F = dataset.get_hmo_feats0()
results = behavioralthor.utils.training_curve(F, dataset, npc_train_list, eval_config_base)
cPickle.dump(results, open('sgd_results.p', 'wb'))