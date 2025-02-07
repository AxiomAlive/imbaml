import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

from config_spaces import MLModelGenerator


class BaggingEnsembleGenerator(MLModelGenerator):
    n_estimators = hp.pchoice('bag.n_estimators', [(0.0625, 8), (0.175, 9), (0.525, 10), (0.175, 11), (0.0625, 12)])
    bootstrap = hp.choice('bag.bootstrap', [True, False])
    max_samples = hp.pchoice('bag.max_samples', [(0.05, 0.8), (0.15, 0.9), (0.8, 1.0)])
    max_features = hp.pchoice('bag.max_features', [(0.05, 0.8), (0.15, 0.9), (0.8, 1.0)])
    bootstrap_features = hp.choice('bag.bootstrap_features', [True, False])

    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': BalancedBaggingClassifier})

        return param_map


class RandomForestGenerator(MLModelGenerator):
    n_estimators = scope.int(hp.loguniform('rf.n_estimators', np.log(9.5), np.log(3000.5)))
    bootstrap = hp.choice('rf.bootstrap', [True, False])
    criterion = hp.choice('rf.criterion', ['gini', 'entropy'])
    max_features = hp.uniform('rf.max_features', 0.05, 1)
    min_samples_split = hp.choice('rf.min_samples_split', [2, 3])
    min_samples_leaf = scope.int(hp.loguniform('rf.min_samples_leaf', np.log(1.5), np.log(50.5)))
    class_weight = hp.choice('rf.class_weight', ['balanced', 'balanced_subsample', None])

    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': RandomForestClassifier})

        return param_map


class BRFGenerator(RandomForestGenerator):
    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': BalancedRandomForestClassifier})

        return param_map