import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

from search_spaces import MLModelGenerator
from search_spaces.balanced.ensemble.bag import RandomForestGenerator


class BalancedBaggingClassifierGenerator(MLModelGenerator):
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


class BalancedRandomForestGenerator(RandomForestGenerator):
    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': BalancedRandomForestClassifier})

        return param_map