import logging
import os
import pprint
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Union, Optional, List, Tuple, final

import numpy as np
import pandas as pd
from imblearn.datasets import make_imbalance
from imblearn.metrics import geometric_mean_score
from sklearn.exceptions import NotFittedError
from sklearn.metrics import fbeta_score, balanced_accuracy_score, recall_score, precision_score, cohen_kappa_score, \
    precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder

from experiment.benchmark import FittedModel, ZenodoExperimentRunner
from utils.decorators import ExceptionWrapper
from sklearn.model_selection import train_test_split as tts

logger = logging.getLogger(__name__)


class AutoMLRunner(ABC):
    def __init__(self, metric):
        self._metric = metric
        self._benchmark_runner = ZenodoExperimentRunner()

        self.__n_evals = 70
        self._fitted_model: FittedModel = None

        self._configure_environment()

    def get_benchmark_runner(self):
        return self._benchmark_runner

    def _configure_environment(self):
        np.random.seed(42)

        logger.info("Prepared env.")


    @abstractmethod
    def fit(
            self,
            X_train: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            target_label: str,
            dataset_name: str,
            n_evals: int):
        raise NotImplementedError()

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self._fitted_model is None:
            raise NotFittedError()

        predictions = self._fitted_model.predict(X_test)
        return predictions

    def _make_imbalance(self, X_train, y_train, class_belongings, pos_label) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
        is_dataset_initially_imbalanced = True
        number_of_positives = class_belongings.get(pos_label)

        proportion_of_positives = number_of_positives / len(y_train)

        # For extreme case - 0.01, for moderate - 0.2, for mild - 0.4.
        if proportion_of_positives > 0.01:
            coefficient = 0.01
            updated_number_of_positives = int(coefficient * len(y_train))
            if len(str(updated_number_of_positives)) < 2:
                logger.warning(f"Number of positive class instances is too low.")
            else:
                class_belongings[pos_label] = updated_number_of_positives
                is_dataset_initially_imbalanced = False

        if not is_dataset_initially_imbalanced:
            X_train, y_train = make_imbalance(
                X_train,
                y_train,
                sampling_strategy=class_belongings)
            logger.info("Imbalancing applied.")

        return X_train, y_train

    @final
    def _log_val_loss_alongside_model_class(self, losses):
        for m, l in losses.items():
            logger.info(f"Validation loss: {float(l):.3f}")
            logger.info(pprint.pformat(f'Model class: {m}'))

    @ExceptionWrapper.log_exception
    def run(self, n_evals: Optional[int] = None):
        if n_evals is not None:
            self.__n_evals = n_evals

        for task in self._benchmark_runner.get_tasks():
            if task is None:
                return

            n_evals = self.__n_evals
            if task.id in [9, 23, 26]:
                n_evals //= 2

            if isinstance(task.X, np.ndarray):
                X_train, X_test, y_train, y_test = self.split_data_on_train_and_test(task.X, task.y)
            elif isinstance(task.X, pd.DataFrame):
                preprocessed_data = self.preprocess_data(task.X, task.y.squeeze())

                if preprocessed_data is None:
                    return

                X, y = preprocessed_data
                X_train, X_test, y_train, y_test = self.split_data_on_train_and_test(X, y.squeeze())
            else:
                raise TypeError(f"pd.DataFrame or np.ndarray expected. Got: {type(task.X)}")

            logger.info(f"{task.id}...Loaded dataset name: {task.name}.")
            logger.info(f'Rows: {X_train.shape[0]}. Columns: {X_train.shape[1]}')

            class_belongings = Counter(y_train)
            logger.info(class_belongings)

            if len(class_belongings) > 2:
                logger.info("Multiclass problems are not currently supported.")
                return

            iterator_of_class_belongings = iter(sorted(class_belongings))
            *_, positive_class_label = iterator_of_class_belongings
            logger.info(f"Pos class label: {positive_class_label}")

            number_of_positives = class_belongings.get(positive_class_label, None)

            if number_of_positives is None:
                logger.error("Unknown positive class label.")
                return

            # X_train, y_train = self._make_imbalance(X_train, y_train, class_belongings, positive_class_label)

            number_of_train_instances_by_class = Counter(y_train)
            logger.info(number_of_train_instances_by_class)

            # estimated_dataset_size_in_memory = y_train.memory_usage(deep=True) / (1024 ** 2)
            # logger.info(f"Dataset size: {estimated_dataset_size_in_memory}")

            start_time = time.time()
            self.fit(X_train, y_train, task.target_label, task.name, n_evals)
            self.examine_quality('time_passed', start_time=start_time)

            y_predictions = self.predict(X_test)
            self.examine_quality(self._metric, y_test, y_predictions, positive_class_label)

    def _compute_metric_score(self, metric: str, *args, **kwargs):
        y_test = kwargs.get("y_test")
        y_pred = kwargs.get("y_pred")
        pos_label = kwargs.get("pos_label")
        start_time = kwargs.get("start_time")

        if metric == 'f1':
            f1 = fbeta_score(y_test, kwargs.get("y_pred"), beta=1, pos_label=pos_label)
            logger.info(f"F1: {f1:.3f}")
        elif metric == 'balanced_acc':
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            logger.info(f"Balanced accuracy: {balanced_accuracy:.3f}")
        elif metric == 'auc_pr':
            pr_curve = precision_recall_curve(y_test, y_pred, pos_label=pos_label)

            auc_pr = auc(pr_curve)
            logger.info(f"AUC-PR: {auc_pr:.3f}")
        elif metric == 'recall':
            recall = recall_score(y_test, y_pred, pos_label=pos_label)
            logger.info(f"Recall: {recall:.3f}")
        elif metric == 'precision':
            precision = precision_score(y_test, y_pred, pos_label=pos_label)
            logger.info(f"Precision: {precision:.3f}")
        elif metric == 'kappa':
            kappa = cohen_kappa_score(y_test, y_pred)
            logger.info(f"Kappa: {kappa:.3f}")
        elif metric == 'time_passed':
            time_passed = time.time() - start_time
            logger.info(f"Time passed: {time_passed // 3600} hours, {time_passed // 60 % 60} minutes and {time_passed % 60} seconds.")

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

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        X.dropna(inplace=True)

        if type(y.iloc[0]) is str:
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y))

        for dataset_feature_name in X.copy():
            dataset_feature = X.get(dataset_feature_name)

            if len(dataset_feature) == 0:
                X.drop([dataset_feature_name], axis=1, inplace=True)
                continue
            if type(dataset_feature.iloc[0]) is str:
                dataset_feature_encoded = pd.get_dummies(dataset_feature, prefix=dataset_feature_name)
                X.drop([dataset_feature_name], axis=1, inplace=True)
                X = pd.concat([X, dataset_feature_encoded], axis=1).reset_index(drop=True)

        if len(X.index) != len(y.index):
            logger.warning(f"X index: {X.index} and y index {y.index}.")
            logger.error("Unexpected X size.")
            return None

        return X, y

    def split_data_on_train_and_test(self, X, y):
        return tts(
            X,
            y,
            random_state=42,
            test_size=0.2,
            stratify=y)
