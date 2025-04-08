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
from lightgbm import LGBMClassifier
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
from xgboost import XGBClassifier

from search_spaces import MLModelGenerator


logger = logging.getLogger(__name__)


class AdaGenerator(MLModelGenerator):
    n_estimators = scope.int(hp.loguniform('ada.n_estimators', np.log(10.5), np.log(500.5)))
    learning_rate = hp.lognormal('ada.learning_rate', np.log(0.01), np.log(20.0))

    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': model_class if model_class is not None else AdaBoostClassifier})

        return param_map

class XGBoostGenerator(MLModelGenerator):
    max_depth = scope.int(hp.uniform('xgb.max_depth', 1, 11))
    learning_rate = hp.loguniform('xgb.learning_rate', np.log(0.0001), np.log(0.5)) - 0.0001
    n_estimators=  scope.int(hp.quniform('xgb.n_estimators', 100, 6000, 200))
    min_child_weight = scope.int(hp.loguniform('xgb.min_child_weight', np.log(1), np.log(100)))
    subsample = hp.uniform('xgb.subsample', 0.5, 1)
    colsample_bylevel =  hp.uniform('xgb.colsample_bylevel', 0.5, 1)
    colsample_bytree =  hp.uniform('xgb.colsample_bytree', 0.5, 1)
    reg_alpha = hp.loguniform('xgb.reg_alpha', np.log(0.0001), np.log(1)) - 0.0001
    reg_lambda = hp.loguniform('xgb.reg_lambda', np.log(1), np.log(4))

    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': XGBClassifier})

        return param_map


class LightGBMGenerator(MLModelGenerator):
    max_depth = scope.int(hp.uniform('lgbm.max_depth', 1, 11))
    learning_rate = hp.loguniform('xgb.learning_rate', np.log(0.0001), np.log(0.5)) - 0.0001
    n_estimators=  scope.int(hp.quniform('xgb.n_estimators', 100, 6000, 200))
    min_child_weight = scope.int(hp.loguniform('xgb.min_child_weight', np.log(1), np.log(100)))
    subsample = hp.uniform('xgb.subsample', 0.5, 1)
    colsample_bytree =  hp.uniform('xgb.colsample_bytree', 0.5, 1)
    reg_alpha = hp.loguniform('xgb.reg_alpha', np.log(0.0001), np.log(1)) - 0.0001
    reg_lambda = hp.loguniform('xgb.reg_lambda', np.log(1), np.log(4))
    num_leaves = scope.int(hp.uniform('lgbm.num_leaves', 2, 121))
    boosting_type = hp.choice('lgbm.boosting_type', ['gbdt', 'dart', 'goss'])

    @classmethod
    def generate_algorithm_configuration_space(cls, model_class=None):
        param_map = super().generate_algorithm_configuration_space()
        param_map.update({'model_class': LGBMClassifier})

        return param_map
