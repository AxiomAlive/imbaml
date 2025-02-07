import math
import multiprocessing
import os
import pprint
import traceback
from os import walk

import arff
import numpy as np
import openml
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

from experiment.runner import BenchmarkExperimentRunner, OpenMLExperimentRunner, ZenodoExperimentRunner
from utils.decorators import ExceptionWrapper
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class AGExperimentRunner(ZenodoExperimentRunner):
    def __init__(self, preset):
        super().__init__()
        self._preset = preset
    # check how to apply decorators for abstract class inheritance case.
    @ExceptionWrapper.log_exception
    def predict(self, X_test):
        if self._fitted_model is None:
            raise NotFittedError()

        dataset_test = TabularDataset(X_test)
        predictions = self._fitted_model.predict(dataset_test)

        return predictions

    @ExceptionWrapper.log_exception
    def fit(self,
            X_train,
            y_train,
            target_label,
            dataset_name
            ):
        if target_label is None and isinstance(X_train, np.ndarray):
            dataset_train = pd.DataFrame(data=np.column_stack([X_train, y_train]))
            autogluon_dataset_train = pd.DataFrame(dataset_train)
            target_label = list(autogluon_dataset_train.columns)[-1]
        else:
            dataset_train = pd.DataFrame(
                data=np.column_stack([X_train, y_train]),
                columns=[*X_train.columns, target_label])

            autogluon_dataset_train = TabularDataset(pd.DataFrame(
                dataset_train,
                columns=[*X_train.columns, target_label]))

        autogluon_predictor = TabularPredictor(
            problem_type='binary',
            label=target_label,
            eval_metric='f1',
            verbosity=2) \
            .fit(
            autogluon_dataset_train,
            presets=[self._preset, 'optimize_for_deployment'],
            num_stack_levels=2
        )

        logger.info(f"Training on dataset {dataset_name} finished.")

        val_scores = autogluon_predictor.leaderboard().get('score_val')
        logger.info(val_scores)
        if len(val_scores) == 0:
            logger.error("No best model found.")
            return

        best_model_name = autogluon_predictor.model_best

        val_losses = {best_model_name: val_scores.max()}
        self._log_val_loss_alongside_model_class(val_losses)

        autogluon_predictor.delete_models(models_to_keep=best_model_name, dry_run=False)

        self._fitted_model = autogluon_predictor
