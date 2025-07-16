import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

from imbaml.search_space import EstimatorSpaceGenerator
from imbaml.search_space.classical.ensemble.bag import RandomForestGenerator


class BalancedBaggingClassifierGenerator(EstimatorSpaceGenerator):
    n_estimators = hp.pchoice('bag.n_estimators', [(0.0625, 8), (0.175, 9), (0.525, 10), (0.175, 11), (0.0625, 12)])
    bootstrap = hp.choice('bag.bootstrap', [True, False])
    max_samples = hp.pchoice('bag.max_samples', [(0.05, 0.8), (0.15, 0.9), (0.8, 1.0)])
    max_features = hp.pchoice('bag.max_features', [(0.05, 0.8), (0.15, 0.9), (0.8, 1.0)])
    bootstrap_features = hp.choice('bag.bootstrap_features', [True, False])

    @classmethod
    def generate(cls, model_class=None):
        param_map = super().generate()
        param_map.update({'model_class': BalancedBaggingClassifier})

        return param_map


class BalancedRandomForestClassifierGenerator(RandomForestGenerator):
    @classmethod
    def generate(cls, model_class=None):
        param_map = super().generate()
        param_map.update({'model_class': BalancedRandomForestClassifier})

        return param_map