import numpy as np
from hyperopt import hp
from hyperopt.pyll import scope
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from imbaml.search_spaces import MLModelGenerator


class RandomForestGenerator(MLModelGenerator):
    n_estimators = scope.int(hp.loguniform('rf.n_estimators', np.log(9.5), np.log(3000.5)))
    bootstrap = hp.choice('rf.bootstrap', [True, False])
    criterion = hp.choice('rf.criterion', ['gini', 'entropy'])
    max_features = hp.pchoice('rf.max_features', [
        (0.2, 'sqrt'),
        (0.1, 'log2'),
        (0.1, None),
        (0.6, hp.uniform('rf.max_features.decimal', 0.0, 1.0))
    ])
    min_samples_split = hp.pchoice('rf.min_samples_split', [(0.95, 2), (0.05, 3)])
    min_samples_leaf = scope.int(hp.loguniform('rf.min_samples_leaf', np.log(1.5), np.log(50.5)))
    class_weight = hp.choice('rf.class_weight', ['balanced', 'balanced_subsample', None])
    max_depth = hp.pchoice('rf.max_depth', [
        (0.7, None),
        (0.1, 2),
        (0.1, 3),
        (0.1, 4)
    ])
    max_leaf_nodes = hp.pchoice('rf.max_leaf_nodes', [
        (0.85, None),
        (0.05, 5),
        (0.05, 10),
        (0.05, 15),
    ])
    min_impurity_decrease = hp.pchoice('rf.min_impurity_decrease', [
        (0.85, 0.0),
        (0.05, 0.01),
        (0.05, 0.02),
        (0.05, 0.05),
    ])
    min_weight_fraction_leaf = 0.0

    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': RandomForestClassifier})

        return param_map

class ExtraTreesGenerator(RandomForestGenerator):
    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': ExtraTreesClassifier})

        return param_map