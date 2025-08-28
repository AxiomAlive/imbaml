import logging
import os
import pprint
import re
from abc import ABC, abstractmethod
from io import StringIO
from typing import Optional, Union, final, List, Dict

import numpy as np
import pandas as pd
import ray
from sklearn.exceptions import NotFittedError
from sklearn.metrics import fbeta_score, balanced_accuracy_score, recall_score, precision_score, average_precision_score

from imbaml.main import ImbamlOptimizer
from utils.decorators import Decorators
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.metrics import make_scorer
from flaml import AutoML as FLAMLPredictor

logger = logging.getLogger(__name__)


class AutoML(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        metric_name: str,
        target_label: Optional[str]
    ) -> None:
        raise NotImplementedError()

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> Union[pd.DataFrame, np.ndarray]:
        if self._fitted_model is None:
            raise NotFittedError()

        predictions = self._fitted_model.predict(X_test)
        return predictions

    @final
    def score(
        self,
        metrics: Union[str, List[str]],
        y_test: Optional[Union[pd.DataFrame, np.ndarray]]=None,
        y_pred: Optional[Union[pd.DataFrame, np.ndarray]]=None,
        pos_label : Optional[int]=None,
    ) -> None:

        calculate_metric_score_kwargs = {
            'y_test': y_test,
            'y_pred': y_pred,
            'pos_label': pos_label
        }

        # TODO: add handling for 'all' as a value of metrics to avoid hard-coding all metric names.
        if isinstance(metrics, str):
            self._calculate_metric_score(
                metrics,
                **calculate_metric_score_kwargs)
        elif isinstance(metrics, list):
            for metric in metrics:
                self._calculate_metric_score(
                    metric,
                    **calculate_metric_score_kwargs)

    @final
    def _log_val_loss_alongside_fitted_model(self, losses: Dict[str, np.float64]) -> None:
        for m, l in losses.items():
            # TODO: different output for leaderboard.
            logger.info(f"Best validation loss: {abs(l):.3f}")

            model_log = pprint.pformat(f"Best model class: {m}", compact=True)
            logger.info(model_log)

    # TODO: persist seed for usage of the same value anywhere.
    def _configure_environment(self, seed=42) -> None:
        np.random.seed(seed)
        logger.info(f"Seed = {seed}.")

    @final
    def _calculate_metric_score(self, metric: str, *args, **kwargs) -> None:
        y_test = kwargs.get("y_test")
        y_pred = kwargs.get("y_pred")
        pos_label = kwargs.get("pos_label")

        if metric == 'f1':
            f1 = fbeta_score(y_test, kwargs.get("y_pred"), beta=1, pos_label=pos_label)
            logger.info(f"F1: {f1:.3f}")
        elif metric == 'balanced_accuracy':
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            logger.info(f"Balanced accuracy: {balanced_accuracy:.3f}")
        elif metric == 'average_precision':
            average_precision = average_precision_score(y_test, y_pred, pos_label=pos_label)
            logger.info(f"Average precision: {average_precision:.3f}")

    def __str__(self):
        return self.__class__.__name__


class Imbaml(AutoML):
    def __init__(
        self,
        sanity_check=False,
        verbosity=0,
        leaderboard=False
    ):
        self._dataset_size: Optional[int] = None
        self._verbosity = verbosity
        if sanity_check:
            self._n_evals = 6
            self._sanity_check = True
        else:
            self._n_evals = 60
            self._sanity_check = False
        self._leaderboard = leaderboard

        super()._configure_environment()
        self._configure_environment()

    @Decorators.log_exception
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        metric_name: str,
        target_label: Optional[str]
    ) -> None:
        n_evals = self._n_evals
        if not self._sanity_check:
            if self._dataset_size is None:
                raise ValueError("Dataset size is undefined.")
            if self._dataset_size > 50:
                n_evals //= 4
            elif self._dataset_size > 5:
                n_evals //= 3

        optimizer = ImbamlOptimizer(
            metric=metric_name,
            re_init=False,
            n_evals=n_evals,
            verbosity=self._verbosity
        )

        fit_results = optimizer.fit(X_train, y_train)

        if self._leaderboard:
            val_losses = {}
            for i, result in enumerate(fit_results):
                if result.error:
                    logger.info(f"Trial #{i} had an error:", result.error)
                    continue
                val_losses[result.metrics['model']] = result.metrics['loss']
            self._log_val_loss_alongside_fitted_model(val_losses)

        best_trial = fit_results.get_best_result(metric='loss', mode='min')

        best_trial_metrics = best_trial.get('metrics')
        if best_trial_metrics is None:
            raise ValueError("Task run failed. No best trial.")

        best_validation_loss = best_trial_metrics.get('loss')

        best_algorithm_configuration = best_trial_metrics.get('config').get('search_configurations')

        best_model_class = best_algorithm_configuration.get('model_class')
        best_algorithm_configuration.pop('model_class')

        best_model = best_model_class(**best_algorithm_configuration)
        model_with_loss = {best_model: best_validation_loss}
        self._log_val_loss_alongside_fitted_model(model_with_loss)

        best_model.fit(X_train, y_train)

        self._fitted_model = best_model

    @property
    def dataset_size(self):
        return self._dataset_size

    @dataset_size.setter
    def dataset_size(self, value):
        self._dataset_size = value

    def _configure_environment(self, seed=42) -> None:
        ray.init(object_store_memory=10**9, log_to_driver=False, logging_level=logging.ERROR)

        os.environ['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'
        os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'
        os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'


class AutoGluon(AutoML):
    def __init__(self, preset='medium_quality', verbosity=0):
        self._preset = preset
        self._verbosity = verbosity

    @property
    def preset(self):
        return self._preset

    @preset.setter
    def preset(self, preset):
        if preset not in ['medium_quality', 'good_quality', 'high_quality', 'best_quality', 'extreme_quality']:
            raise ValueError(
                """
                Invalid value of parameter preset.
                Options available: ['medium_quality', 'good_quality', 'high_quality', 'best_quality', 'extreme_quality'].
                """)
        self._preset = preset

    @Decorators.log_exception
    def predict(self, X_test: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        if self._fitted_model is None:
            raise NotFittedError()

        dataset_test = TabularDataset(X_test)
        predictions = self._fitted_model.predict(dataset_test)

        return predictions

    @Decorators.log_exception
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        metric_name: str,
        target_label: Optional[str]
    ) -> None:
        if metric_name not in ['f1', 'balanced_accuracy', 'average_precision']:
            raise ValueError(f"Metric {metric_name} is not supported.")

        if target_label is None and isinstance(X_train, np.ndarray):
            Xy_train = pd.DataFrame(data=np.column_stack([X_train, y_train]))
            target_label = list(Xy_train.columns)[-1]
        else:
            Xy_train = pd.DataFrame(
                data=np.column_stack([X_train, y_train]),
                columns=[*X_train.columns, target_label])

        dataset = TabularDataset(Xy_train)
        predictor = TabularPredictor(
            problem_type='binary',
            label=target_label,
            eval_metric=metric_name,
            verbosity=self._verbosity
        ).fit(dataset)

        val_scores = predictor.leaderboard().get('score_val')
        if val_scores is None or len(val_scores) == 0:
            logger.error("No best model found.")
            return

        best_model_name = predictor.model_best

        val_losses = {best_model_name: np.float64(val_scores.max())}
        self._log_val_loss_alongside_fitted_model(val_losses)

        predictor.delete_models(models_to_keep=best_model_name, dry_run=False)

        self._fitted_model = predictor


class FLAML(AutoML):
    def __init__(self):
        pass

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series], metric_name: str,
            target_label: Optional[str]) -> None:
        metric = None
        if metric_name == 'average_precision':
            metric = 'ap'
        elif metric_name == 'f1':
            metric = 'f1'
        elif metric_name in ['balanced_accuracy', 'precision', 'recall']:
            raise ValueError(f"Metric {metric_name} is not supported.")

        automl = FLAMLPredictor()
        automl.fit(X_train, y_train, metric=metric)

        best_loss = automl.best_loss
        best_model = automl.best_estimator

        self._log_val_loss_alongside_fitted_model({best_model: np.float64(best_loss)})

        self._fitted_model = automl
