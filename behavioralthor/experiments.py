__author__ = 'headradio'
from devthor.procedures import suggest_multiple_from_name


def fruits_vs_chairs(dbname='fruits_vs_chairs', randomSearch=False):
    host = 'localhost'
    port = 22334
    bandit_names = ['behavioralthor.bandits.object_level_fruits_and_chairs_L3',
                    # 'behavioralthor.bandits.fruits_vs_chairs_L3',
                    'behavioralthor.bandits.fruits_vs_chairs_L2',
                    'behavioralthor.bandits.fruits_vs_chairs_L4',
                    'behavioralthor.bandits.fruits_vs_chairs_L1R']
    nExps = len(bandit_names)
    bandit_args_list = [() for _i in range(nExps)]
    bandit_kwargs_list = [{} for _i in range(nExps)]
    exp_keys = ['obj_L3',
                # 'L3',
                'L2',
                'L4',
                'L1R']
    if randomSearch:
        bandit_algo_names = ['hyperopt.Random'] * nExps
        bandit_algo_args_list = [() for _i in range(nExps)]
        bandit_algo_kwargs_list = [{} for _i in range(nExps)]
    else:
        bandit_algo_names = ['hyperopt.TreeParzenEstimator'] * nExps
        bandit_algo_args_list = [() for _i in range(nExps)]
        bandit_algo_kwargs_list = [{'gamma': 0.25, 'n_startup_jobs': 100} for _i in range(nExps)]

    N = None
    return suggest_multiple_from_name(
        dbname=dbname,
        host=host,
        port=port,
        bandit_algo_names=bandit_algo_names,
        bandit_names=bandit_names,
        exp_keys=exp_keys,
        N=None,
        bandit_args_list=bandit_args_list,
        bandit_kwargs_list=bandit_kwargs_list,
        bandit_algo_args_list=bandit_algo_args_list,
        bandit_algo_kwargs_list=bandit_algo_kwargs_list
    )