import logging
from typing import Union

import pandas    as pd
import numpy as np

from hyperopt import STATUS_OK, hp

from ray.tune import Tuner
from ray.tune.search import ConcurrencyLimiter
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imbens.ensemble import AdaCostClassifier
from sklearn.metrics import *

from search_spaces.imbalanced.ensemble.boost import AdaReweightedGenerator
from search_spaces.balanced.ensemble.boost import XGBoostGenerator, LightGBMGenerator
from search_spaces.balanced.ensemble.bag import ExtraTreesGenerator
from search_spaces.imbalanced.ensemble.bag import BalancedBaggingClassifierGenerator, BalancedRandomForestGenerator
from utils.decorators import ExceptionWrapper

from ray.tune.search.hyperopt import HyperOptSearch
from ray.train import RunConfig
import ray

from .runner import AutoMLRunner
from imbaml.main import Imba


logger = logging.getLogger(__name__)


class ImbaExperimentRunner(AutoMLRunner):
    def __init__(self, metrics):
        super()._configure_environment()
        super().__init__(metrics)

    def _configure_environment(self):
        ray.init(object_store_memory=10**9, log_to_driver=False, logging_level=logging.ERROR)

    @ExceptionWrapper.log_exception
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        metric_name: str,
        target_label: str,
        dataset_name: str
    ) -> None:
        automl = Imba(metric=metric_name, re_init=False)

        fit_results = automl.fit(X_train, y_train)

        best_trial = fit_results.get_best_result(metric='loss', mode='min')
        if best_trial is None:
            raise ValueError("No best trial.")

        best_trial_metrics = getattr(best_trial, 'metrics')
        if best_trial_metrics is None:
            raise ValueError("No best trial metrics.")

        best_validation_loss = best_trial_metrics.get('loss')
        if best_validation_loss is None:
            raise ValueError("No best trial validation loss.")

        best_algorithm_configuration = best_trial_metrics.get('config').get('search_configurations')
        if best_algorithm_configuration is None:
            raise ValueError("No best trial algorithm configuration.")

        best_model_class = best_algorithm_configuration.get('model_class')
        if best_model_class is None:
            raise ValueError("No best trial model class.")

        best_algorithm_configuration.pop('model_class')
        best_model = best_model_class(**best_algorithm_configuration)

        val_losses = {best_model: best_validation_loss}
        self._log_val_loss_alongside_model_class(val_losses)

        best_model.fit(X_train, y_train)

        self._fitted_model = best_model

    @ExceptionWrapper.log_exception
    def predict(self, X_test):
        if self._fitted_model is None:
            raise NotFittedError()

        predictions = self._fitted_model.predict(X_test)
        return predictions
