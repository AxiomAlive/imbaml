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
from abc import ABC, abstractmethod
import numpy as np
from hyperopt import hp
from typing import TypeVar
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from loguru import logger

from imbaml.search_space import EstimatorSpaceGenerator
from imbaml.search_space.classical.ensemble.boost import AdaBoostClassifierGenerator


class AdaReweightedGenerator(AdaBoostClassifierGenerator):
    early_termination = True

    @classmethod
    def generate(cls, model_class=None):
        if model_class is None:
            raise ValueError("Model class must be specified for AdaReweightedGenerator.")
        param_map = super().generate()
        param_map.update({'model_class': model_class})

        return param_map


class RUSBoostGenerator(AdaBoostClassifierGenerator):
    @classmethod
    def generate(cls, model_class=None):
        param_map = super().generate()
        param_map.update({'model_class': RUSBoostClassifier})

        return param_map


class EasyEnsembleGenerator(EstimatorSpaceGenerator):
    n_estimators = scope.int(hp.loguniform('easy_ensemble.n_estimators', np.log(10.5), np.log(500.5)))

    @classmethod
    def generate(cls, model_class=None):
        param_map = super().generate()
        param_map.update({'model_class': EasyEnsembleClassifier})

        return param_map
