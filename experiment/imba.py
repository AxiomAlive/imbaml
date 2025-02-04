import logging
import pprint
import traceback
from abc import abstractmethod, ABC
from typing import Optional, Tuple, Union, Any, TypeVar

import arff
import openml
import pandas    as pd
import numpy as np
from collections import Counter

import sklearn.metrics
from hyperopt import fmin, tpe, STATUS_OK, Trials, rand, hp, space_eval
from functools import partial

from ray.tune import Tuner
from ray.tune.search import ConcurrencyLimiter
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split as tts, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.datasets import fetch_datasets, make_imbalance
from imbens.ensemble import AdaUBoostClassifier, AdaCostClassifier, AsymBoostClassifier, OverBoostClassifier, \
    SMOTEBoostClassifier, OverBaggingClassifier, BalancedRandomForestClassifier
from os import walk
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from config_spaces.ada import AdaReweightedGenerator
from config_spaces.forest import RandomForestGenerator, BRFGenerator
from utils.decorators import ExceptionWrapper
from .runner import ExperimentRunner, OpenMLExperimentRunner, ZenodoExperimentRunner

import ray
from ray.tune.search.hyperopt import HyperOptSearch


logger = logging.getLogger(__name__)


class RayTrainable(ray.tune.Trainable):
    def setup(self, config):
        self.algorithm_configuration = config["algorithm_configuration"]
        self.metric = config["metric"]
        self.X = config['X']
        self.y = config['y']

    def step(self):
        trial_result = ImbaExperimentRunner.compute_metric_score(
            self.algorithm_configuration,
            self.metric,
            self.X,
            self.y)
        return {"loss": trial_result['loss']}


class ImbaExperimentRunner(ZenodoExperimentRunner):
    @classmethod
    def compute_metric_score(cls, hyper_parameters, metric, X, y):
        hyper_parameters = hyper_parameters.copy()
        model_class = hyper_parameters.pop('model_class')
        clf = model_class(**hyper_parameters)

        loss_value = cross_val_score(
            estimator=clf,
            X=X,
            y=y,
            cv=StratifiedKFold(n_splits=3),
            scoring=make_scorer(metric, pos_label=1),
            error_score='raise').mean()

        return {'loss': loss_value, 'status': STATUS_OK}

    @ExceptionWrapper.log_exception
    def fit(
            self,
            X_train: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            target_label: str,
            dataset_name: str
            ):

        number_of_instances = X_train.shape[0]
        number_of_columns = X_train.shape[1]
        logger.info(f'N = {number_of_instances}. M = {number_of_columns}')

        model_classes = [
                AdaReweightedGenerator.generate_algorithm_configuration_space(AdaUBoostClassifier),
                AdaReweightedGenerator.generate_algorithm_configuration_space(AdaCostClassifier),
                AdaReweightedGenerator.generate_algorithm_configuration_space(AsymBoostClassifier),
                BRFGenerator.generate_algorithm_configuration_space()
            ]

        algorithms_configuration = hp.choice("algorithm_configuration", model_classes)
        ray_configuration = {
            'X': X_train,
            'y': y_train,
            'metric': f1_score,
            'algorithm_configuration': algorithms_configuration
        }

        # HyperOptSearch(points_to_evaluate = promising initial points)
        search_algo = ConcurrencyLimiter(
            HyperOptSearch(
                space=ray_configuration,
                metric='loss',
                mode='min'),
            max_concurrent=4,
            batch=True)

        tuner = ray.tune.Tuner(
            RayTrainable,
            tune_config=ray.tune.TuneConfig(
                metric='loss',
                mode='max',
                search_alg=search_algo,
                num_samples=self._n_evals,
                reuse_actors=True),
            # param_space=ray_configuration
        )

        logger.info(f"Document {dataset_name}")

        results = tuner.fit()

        best_trial = results.get_best_result(metric='loss', mode='min')

        best_trial_metrics = getattr(best_trial, 'metrics')
        if best_trial_metrics is None:
            raise Exception("Optimization failed. No best trial found.")

        logger.info(f"Training on dataset {dataset_name} finished.")

        best_validation_loss = best_trial_metrics.get('loss')

        best_algorithm_configuration = best_trial_metrics.get('config').get('algorithm_configuration')

        best_model_class = best_algorithm_configuration.get('model_class')
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
