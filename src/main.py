import logging
from typing import Union, Callable

import numpy as np
import pandas as pd
import ray
from hyperopt import hp, STATUS_OK
from ray.tune import ResultGrid
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import make_scorer, f1_score, balanced_accuracy_score, average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import *
from imblearn.ensemble import *
from imbens.ensemble import *
from sklearn.neural_network import *

logger = logging.getLogger(__name__)


# TODO: redesign to inherit from ray.tune.Trainable.
class RayTuner:
    @staticmethod
    def trainable(config):
        trial_result = ImbamlOptimizer.compute_metric_score(
            config['search_configurations'],
            config['metric'],
            config['X'],
            config['y'])
        ray.train.report(trial_result)        


class ImbamlOptimizer:
    def __init__(self, metric, n_evals, verbosity=0, re_init=True):
        self._metric = metric
        self._n_evals = n_evals
        self._verbosity = verbosity
        if re_init:
            ray.init(object_store_memory=10**9, log_to_driver=False, logging_level=logging.ERROR)
                
    @classmethod
    def compute_metric_score(cls, hyper_parameters, metric, X, y):
        hyper_parameters = hyper_parameters.copy()
        model_class = hyper_parameters.pop('model_class')
        clf = model_class(**hyper_parameters)

        loss_value = cross_val_score(
            estimator=clf,
            X=X,
            y=y,
            cv=StratifiedKFold(n_splits=8),
            scoring=make_scorer(metric),
            error_score='raise'
        ).mean()

        return {
            'loss': -loss_value,
            'model': str(model_class(**hyper_parameters)),
            'status': STATUS_OK
        }

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> ResultGrid:
        metric: Callable
        if self._metric == 'f1':
            metric = f1_score
        elif self._metric == 'balanced_accuracy':
            metric = balanced_accuracy_score
        elif self._metric == 'average_precision':
            metric = average_precision_score
        else:
            raise ValueError(f"Metric {self._metric} is not supported.")

        # TODO: fix - AdaReweighted family produces a bunch of erroneous trials.
        # search_space = [
            # XGBClassifierGenerator.generate(),
            # AdaReweightedGenerator.generate(AdaCostClassifier),
            # BalancedRandomForestClassifierGenerator.generate(),
            # BalancedBaggingClassifierGenerator.generate(),
            # LGBMClassifierGenerator.generate(),
            # ExtraTreesGenerator.generate(),
            # MLPClassifierGenerator.generate()
        # ]

        estimators = [
            ("xgb", XGBClassifier()),
            ("ada", AdaCostClassifier()),
            ("rf", BalancedRandomForestClassifier()),
            ("bag", BalancedBaggingClassifier()),
            ("lgbm", LGBMClassifier()),
            ("extra", ExtraTreesClassifier()),
            ("mlp", MLPClassifier()),
        ]
        search_space = StackingClassifier(estimators)
        logger.info(f"Search space: {search_space}")

        search_configurations = hp.choice("search_configurations", search_space)

        ray_configuration = {
            'X': X,
            'y': y,
            'metric': metric,
            'search_configurations': search_configurations
        }

        # HyperOptSearch(points_to_evaluate = promising initial points)
        search_algo = ConcurrencyLimiter(
            HyperOptSearch(
                space=ray_configuration,
                metric='loss',
                mode='min'),
            max_concurrent=4)

        scheduler = ASHAScheduler(reduction_factor=3)

        # TODO: Consider reusage of actors.
        tuner = ray.tune.Tuner(
            RayTuner.trainable,
            tune_config=ray.tune.TuneConfig(
                metric='loss',
                mode='min',
                search_alg=search_algo,
                num_samples=self._n_evals,
                scheduler=scheduler),
            run_config=ray.train.RunConfig(
                verbose=self._verbosity
            )
        )

        return tuner.fit()
