import logging
import pprint
from typing import Tuple, List, Optional, Union

import numpy as np
import pandas as pd
import timeit

from hyperopt.pyll import scope
from imbens.ensemble import AdaCostClassifier, AsymBoostClassifier
from imbens.ensemble.reweighting import AdaUBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier, EasyEnsembleClassifier
from matplotlib import pyplot as plt
from openml import flows as openml_flows
from openml import runs as openml_runs
from openml import tasks as openml_tasks
from openml import utils as openml_utils
from abc import ABC, abstractmethod
import numpy as np
from hyperopt import hp
from typing import TypeVar
from sklearn.ensemble import AdaBoostClassifier
from config_spaces import MLModelGenerator


logger = logging.getLogger(__name__)


class AdaGenerator(MLModelGenerator):
    n_estimators = scope.int(hp.loguniform('ada.n_estimators', np.log(10.5), np.log(500.5)))
    learning_rate = hp.lognormal('ada.learning_rate', np.log(0.01), np.log(20.0))

    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': model_class if model_class is not None else AdaBoostClassifier})

        return param_map


class AdaReweightedGenerator(AdaGenerator):
    early_termination = True

    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        if model_class is None:
            raise ValueError("Model class must be specified for AdaReweightedGenerator.")
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': model_class})

        return param_map


class RUSBoostGenerator(AdaGenerator):
    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': RUSBoostClassifier})

        return param_map


class EasyEnsembleGenerator(MLModelGenerator):
    n_estimators = scope.int(hp.loguniform('easy_ensemble.n_estimators', np.log(10.5), np.log(500.5)))

    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': EasyEnsembleClassifier})

        return param_map