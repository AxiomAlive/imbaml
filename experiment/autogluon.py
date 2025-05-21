import math
import multiprocessing
import os
import pprint
import traceback
from os import walk
from typing import Union

import arff
import numpy as np
import pandas as pd
import sklearn.metrics
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from sklearn.exceptions import NotFittedError
from sklearn.metrics import *
from sklearn.model_selection import train_test_split as tts
from imblearn.datasets import make_imbalance
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import torch
import logging
import pickle

from experiment.repository import BenchmarkExperimentRunner, OpenMLExperimentRunner, ZenodoExperimentRunner
from experiment.runner import AutoMLRunner
from utils.decorators import ExceptionWrapper
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class AutoGluonExperimentRunner(AutoMLRunner):
    def __init__(self, metrics, preset='good_quality'):
        super().__init__(metrics)
        self._preset = preset

    def get_preset(self):
        return self._preset

    def set_preset(self, preset):
        if preset not in ['medium_quality', 'good_quality', 'high_quality', 'best_quality']:
            raise ValueError(
                """
                Invalid value of parameter preset.
                Options available: ['medium_quality', 'good_quality', 'high_quality', 'best_quality'].
                """)
        self._preset = preset

    # TODO: check how to apply decorators for abstract class inheritance case.
    @ExceptionWrapper.log_exception
    def predict(self, X_test) -> Union[pd.Series, np.ndarray]:
        if self._fitted_model is None:
            raise NotFittedError()

        dataset_test = TabularDataset(X_test)
        predictions = self._fitted_model.predict(dataset_test)

        return predictions

    @ExceptionWrapper.log_exception
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        metric_name: str,
        target_label: str,
        dataset_name: str
    ) -> None:
        if metric_name  in ['f1', 'balanced_accuracy', 'average_precision', 'recall', 'precision']:
            self._metric_automl_arg = metric_name
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
            eval_metric=self._metric_automl_arg,
            verbosity=2) \
            .fit(
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
