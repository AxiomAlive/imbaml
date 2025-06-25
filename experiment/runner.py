import logging
import pprint
import time
from abc import ABC, abstractmethod
from collections import Counter
from typing import Union, Optional, List, final

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import fbeta_score, balanced_accuracy_score, recall_score, precision_score, average_precision_score

from common.preprocessing import DatasetPreprocessor
from experiment.repository import FittedModel, ZenodoExperimentRunner
from utils.decorators import Decorators

logger = logging.getLogger(__name__)


class AutoMLExperimentRunner(ABC):
    def __init__(self, metrics):
        self._metrics = metrics
        self._benchmark_runner = ZenodoExperimentRunner()
        self._n_evals = 60
        self._fitted_model: FittedModel = None
        self._configure_environment()

    @property
    def benchmark_runner(self):
        return self._benchmark_runner

    def _configure_environment(self):
        np.random.seed(42)
        logger.info("Set seed to 42.")
        logger.info("Prepared environment.")

    @abstractmethod
    def fit(
            self,
            X_train: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            metric_name: str,
            target_label: str,
            dataset_name: str):
        raise NotImplementedError()

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self._fitted_model is None:
            raise NotFittedError()

        predictions = self._fitted_model.predict(X_test)
        return predictions

    @final
    def _log_val_loss_alongside_model_class(self, losses):
        for m, l in losses.items():
            logger.info(f"Validation loss: {abs(float(l)):.3f}")
            logger.info(pprint.pformat(f'Model class: {m}'))

    @Decorators.log_exception
    def run(self) -> None:
        for task in self._benchmark_runner.get_tasks():
            if task is None:
                return

            if isinstance(task.X, np.ndarray) or isinstance(task.X, pd.DataFrame):
                preprocessor = DatasetPreprocessor()
                preprocessed_data = preprocessor.preprocess_data(task.X, task.y.squeeze())

                assert preprocessed_data is not None

                X, y = preprocessed_data
                X_train, X_test, y_train, y_test = preprocessor.split_data_on_train_and_test(X, y.squeeze())
            else:
                raise TypeError(f"pd.DataFrame or np.ndarray expected. Got: {type(task.X)}")

            logger.info(f"{task.id}...Loaded dataset name: {task.name}.")
            logger.info(f'Rows: {X_train.shape[0]}. Columns: {X_train.shape[1]}')

            class_belongings = Counter(y_train)
            logger.info(class_belongings)

            if len(class_belongings) > 2:
                raise ValueError("Multiclass problems currently not supported.")

            iterator_of_class_belongings = iter(sorted(class_belongings))
            *_, positive_class_label = iterator_of_class_belongings
            logger.info(f"Positive class label: {positive_class_label}")

            number_of_positives = class_belongings.get(positive_class_label)

            if number_of_positives is None:
                raise ValueError("Unknown positive class label.")

            number_of_train_instances_by_class = Counter(y_train)
            logger.info(number_of_train_instances_by_class)

            for metric in self._metrics:
                start_time = time.time()
                self.fit(X_train, y_train, metric, task.target_label, task.name)
                logger.info(f"Training on dataset (id={task.id}, name={task.name}) successfully finished.")

                self.examine_quality('time_passed', start_time=start_time)

                y_predictions = self.predict(X_test)
                self.examine_quality(metric, y_test, y_predictions, positive_class_label)

    def _compute_metric_score(self, metric: str, *args, **kwargs):
        y_test = kwargs.get("y_test")
        y_pred = kwargs.get("y_pred")
        pos_label = kwargs.get("pos_label")
        start_time = kwargs.get("start_time")

        if metric == 'f1':
            f1 = fbeta_score(y_test, kwargs.get("y_pred"), beta=1, pos_label=pos_label)
            logger.info(f"F1: {f1:.3f}")
        elif metric == 'balanced_accuracy':
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            logger.info(f"Balanced accuracy: {balanced_accuracy:.3f}")
        elif metric == 'average_precision':
            average_precision = average_precision_score(y_test, y_pred, pos_label=pos_label)
            logger.info(f"Average precision: {average_precision:.3f}")
        elif metric == 'recall':
            recall = recall_score(y_test, y_pred, pos_label=pos_label)
            logger.info(f"Recall: {recall:.3f}")
        elif metric == 'precision':
            precision = precision_score(y_test, y_pred, pos_label=pos_label)
            logger.info(f"Precision: {precision:.3f}")
        elif metric == 'time_passed':
            time_passed = time.time() - start_time
            logger.info(f"Time passed: {time_passed // 60} minutes.")

    def examine_quality(
            self,
            metrics: Union[str, List[str]],
            y_test: Optional[Union[pd.DataFrame, np.ndarray]]=None,
            y_pred: Optional[Union[pd.DataFrame, np.ndarray]]=None,
            pos_label:Optional[int]=None,
            start_time:Optional[float]=None):

        compute_metric_score_kwargs = {
            'y_test': y_test,
            'y_pred': y_pred,
            'pos_label': pos_label,
            'start_time': start_time
        }

        # TODO: add handling for 'all' as a value of metrics to avoid hard-coding all metric names.
        if isinstance(metrics, str):
            self._compute_metric_score(
                metrics,
                **compute_metric_score_kwargs)
        elif isinstance(metrics, list):
            for metric in metrics:
                self._compute_metric_score(
                    metric,
                    **compute_metric_score_kwargs)

