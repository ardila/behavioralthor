__author__ = 'headradio'
from simffa import simffa_experiments


def fruits_vs_chairs(dbname='fruits_vs_chairs', randomSearch=False):
    host = 'localhost'
    port = 22334
    bandit_names = ['behavioralthor.bandits.fruits_vs_chairs_L3']
    nExps = len(bandit_names)
    bandit_args_list = [() for _i in range(nExps)]
    bandit_kwargs_list = [{} for _i in range(nExps)]
    exp_keys = ['L3']
    if randomSearch:
        bandit_algo_names = ['hyperopt.Random'] * nExps
        bandit_algo_args_list = [() for _i in range(nExps)]
        bandit_algo_kwargs_list = [{} for _i in range(nExps)]
    else:
        bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * nExps
        bandit_algo_args_list = [() for _i in range(nExps)]
        bandit_algo_kwargs_list = [{'gamma': 0.25, 'n_startup_jobs': 100} for _i in range(nExps)]

    N = None
    exp = simffa_experiments.suggest_multiple_from_name(dbname, host, port, bandit_algo_names, bandit_names,
                                                        exp_keys, N, bandit_args_list, bandit_kwargs_list,
                                                        bandit_algo_args_list, bandit_algo_kwargs_list)
    return exp