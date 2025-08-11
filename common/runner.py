import logging
import pprint
import time
from abc import ABC, abstractmethod
from collections import Counter
from io import StringIO
from typing import Union, Optional, List, final

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from common.automl import Imbaml, AutoGluon, FLAML
from common.preprocessing import DatasetPreprocessor
from benchmark.repository import FittedModel, ZenodoRepository
from utils.decorators import Decorators

logger = logging.getLogger(__name__)


class AutoMLRunner(ABC):
    def __init__(self, automl='imbaml', *args, **kwargs):
        if automl == 'imbaml':
            self._automl = Imbaml(*args, **kwargs)
        elif automl == 'ag':
            self._automl = AutoGluon(*args, **kwargs)
        elif automl == 'flaml':
            self._automl = FLAML()
        else:
            raise ValueError(
                """
                Invalid --automl option.
                Options available: ['imbaml', 'ag', 'flaml'].
                """)

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError()

    @final
    def _run_on_task(self, task: Union[pd.DataFrame, np.ndarray]) -> None:
        if task is None:
            logger.error("Task run failed.")
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
        logger.info(f"Inferred positive class label: {positive_class_label}.")

        number_of_positives = class_belongings.get(positive_class_label)

        if number_of_positives is None:
            raise ValueError("Unknown positive class label.")

        number_of_train_instances_by_class = Counter(y_train)
        logger.info(number_of_train_instances_by_class)

        for metric in self._metrics:
            start_time = time.time()
            self._automl.fit(X_train, y_train, metric, task.target_label, task.name)
            logger.info(f"Training on dataset (id={task.id}, name={task.name}) successfully finished.")

            time_passed = time.time() - start_time
            logger.info(f"Time passed: {time_passed // 60} minutes.")

            y_predictions = self._automl.predict(X_test)
            self._automl.score(metric, y_test, y_predictions, positive_class_label)


class AutoMLSingleRunner(AutoMLRunner):
    def __init__(self, task: Union[pd.DataFrame, np.ndarray], metrics: Optional[List[str]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = metrics
        self._fitted_model: FittedModel = None
        self._task = task

    @Decorators.log_exception
    def run(self) -> None:
        self._run_on_task(self._task)


class AutoMLBenchmarkRunner(AutoMLRunner):
    def __init__(self, metrics: Optional[List[str]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = metrics
        self._repository = ZenodoRepository()
        self._fitted_model: FittedModel = None

    @property
    def repository(self):
        return self._repository

    @Decorators.log_exception
    def run(self) -> None:
        for task in self._repository.get_tasks():
            self._run_on_task(task)

