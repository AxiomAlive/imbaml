import logging
from typing import Union

import pandas    as pd
import numpy as np

from hyperopt import STATUS_OK, hp

from sklearn.exceptions import NotFittedError

from utils.decorators import Decorators

from ray.tune.search.hyperopt import HyperOptSearch
import ray

from .runner import AutoMLExperimentRunner
from imbaml.main import Imba


logger = logging.getLogger(__name__)


class ImbaRunner(AutoMLExperimentRunner):
    """
    ImbaRunner is a class that extends AutoMLExperimentRunner to perform automated machine learning pipeline design using Imba framework.

    Attributes:
        _n_evals (int): The number of evaluations to perform during fitting.
        _fitted_model (object): The model that has been fitted.

    Parameters:
        metrics (list): A list of metrics to evaluate the models.
        is_sanity_check (bool): A flag indicating whether to perform a sanity check, default is False.

    Methods:
        fit(X_train, y_train, metric_name, target_label, dataset_name):
            Fits the model to the training data and logs validation loss alongside the model class.
        
        predict(X_test):
            Predicts the target values for the given test data using the fitted model.
    """
    def __init__(self, metrics, is_sanity_check=False):
        super()._configure_environment()
        super().__init__(metrics)

        if is_sanity_check:
            self._n_evals = 12

    def _configure_environment(self):
        ray.init(object_store_memory=10**9, log_to_driver=False, logging_level=logging.ERROR)

    @Decorators.log_exception
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        metric_name: str,
        target_label: str,
        dataset_name: str
    ) -> None:
        automl = Imba(metric=metric_name, re_init=False, n_evals=self._n_evals)

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

    @Decorators.log_exception
    def predict(self, X_test):
        if self._fitted_model is None:
            raise NotFittedError()

        predictions = self._fitted_model.predict(X_test)
        return predictions
