import logging
import os
import pprint
from abc import ABC, abstractmethod
from io import StringIO
from typing import Optional, Union, final, List

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
        target_label: Optional[str],
        dataset_name: str
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
        pos_label:Optional[int]=None,
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
    def _log_val_loss_alongside_model_class(self, losses):
        for m, l in losses.items():
            logger.info(f"Validation loss: {abs(float(l)):.3f}")

            string_buffer = StringIO()
            pprint.pprint(f'Model class: {m}', string_buffer, compact=True)
            logger.info(string_buffer.getvalue())

    #TODO: persist seed for usage of the same value anywhere.
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


class Imbaml(AutoML):
    def __init__(
        self,
        sanity_check=False,
        verbosity=0,
    ):
        self._verbosity = verbosity
        if sanity_check:
            self._n_evals = 6
            self._sanity_check = True
        else:
            self._n_evals = 60
            self._sanity_check = False

        super()._configure_environment()
        self._configure_environment()

    def _configure_environment(self, seed=42) -> None:
        ray.init(object_store_memory=10**9, log_to_driver=False, logging_level=logging.ERROR)

        os.environ['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'
        os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'
        os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'

    @Decorators.log_exception
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        metric_name: str,
        target_label: Optional[str],
        dataset_name: str
    ) -> None:
        automl = ImbamlOptimizer(
            metric=metric_name,
            re_init=False,
            n_evals=self._n_evals,
            verbosity=self._verbosity,
            sanity_check=self._sanity_check
        )

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


class AutoGluon(AutoML):
    def __init__(self, preset='medium_quality'):
        self._preset = preset

    def get_preset(self):
        return self._preset

    def set_preset(self, preset):
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
        target_label: Optional[str],
        dataset_name: str
    ) -> None:
        if metric_name  in ['f1', 'balanced_accuracy', 'average_precision', 'recall', 'precision']:
            self._metric = metric_name
        else:
            raise ValueError(f"Metric {metric_name} is not supported.")

        if target_label is None and isinstance(X_train, np.ndarray):
            dataset_train = pd.DataFrame(data=np.column_stack([X_train, y_train]))
            autogluon_dataset_train = pd.DataFrame(dataset_train)
            target_label = list(autogluon_dataset_train.columns)[-1]
        else:
            dataset_train = pd.DataFrame(
                data=np.column_stack([X_train, y_train]),
                columns=[*X_train.columns, target_label])

            dataset_train2 = pd.DataFrame(
                dataset_train,
                columns=[*X_train.columns, target_label])
            autogluon_dataset_train = TabularDataset(dataset_train)
            logger.info(dataset_train.equals(dataset_train2))

        autogluon_predictor = TabularPredictor(
            problem_type='binary',
            label=target_label,
            eval_metric=self._metric,
            verbosity=self._verbosity
        ).fit(
            autogluon_dataset_train,
            presets=[self._preset],
        )

        logger.info(f"Training on dataset {dataset_name} finished.")

        val_scores = autogluon_predictor.leaderboard().get('score_val')
        if len(val_scores) == 0:
            logger.error("No best model found.")
            return

        best_model_name = autogluon_predictor.model_best

        val_losses = {best_model_name: val_scores.max()}
        self._log_val_loss_alongside_model_class(val_losses)

        autogluon_predictor.delete_models(models_to_keep=best_model_name, dry_run=False)

        self._fitted_model = autogluon_predictor


class FLAML(AutoML):
    def __init__(self):
        pass

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        metric_name: str,
        target_label: Optional[str],
        dataset_name: str
    ) -> None:

        if metric_name == 'average_precision':
            self._metric = 'ap'
        elif metric_name == 'f1':
            self._metric = 'f1'
        elif metric_name in ['balanced_accuracy', 'precision', 'recall']:
            raise ValueError(f"Metric {metric_name} is not supported.")

        automl = FLAMLPredictor()
        automl.fit(X_train, y_train, task='classification', metric=self._metric)

        best_loss = automl.best_loss
        best_model = automl.best_estimator
        self._log_val_loss_alongside_model_class({best_model: best_loss})

        self._fitted_model = automl
