import logging

from imbaml.search_space import EstimatorSpaceGenerator

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import *
from imblearn.ensemble import *
from imbens.ensemble import *
from sklearn.neural_network import *


logger = logging.getLogger(__name__)


class StackingClassifierGenerator(EstimatorSpaceGenerator):
    estimators = [
        ("mlp", MLPClassifier()),
        ("extra", ExtraTreesClassifier()),
        ("lgbm", LGBMClassifier()),
        ("bag", BalancedBaggingClassifier()),
        ("rf", BalancedRandomForestClassifier()),
        ("ada", AdaCostClassifier()),
        ("xgb", XGBClassifier()),
    ]

    @classmethod
    def generate(cls, model_class=None):
        param_map = super().generate()
        param_map.update({'model_class': StackingClassifier})

        return param_map