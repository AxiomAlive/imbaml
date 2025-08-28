import logging
import pprint
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Union, Optional, List, final

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from common.automl import Imbaml, AutoGluon, FLAML
from common.domain import Dataset
from common.preprocessing import DatasetPreprocessor
from benchmark.repository import FittedModel, ZenodoRepository, DatasetRepository
from utils.decorators import Decorators

logger = logging.getLogger(__name__)


class AutoMLRunner(ABC):
    def __init__(self, automl='imbaml', log_to_file=True, *args, **kwargs):
        self._fitted_model: Optional[FittedModel]
        self._log_to_file = log_to_file
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

    def _configure_environment(self) -> None:
        logging_handlers = [
            logging.StreamHandler(stream=sys.stdout),
        ]

        if self._log_to_file:
            log_filepath = 'logs/'
            Path(log_filepath).mkdir(parents=True, exist_ok=True)
            log_filepath += datetime.now().strftime(f'{self._automl} {",".join(self._metrics)} %Y.%m.%d %H:%M:%S')
            log_filepath += '.log'
            logging_handlers.append(logging.FileHandler(filename=log_filepath, encoding='utf-8', mode='w'))

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=logging_handlers
        )

        logger.info(f"Optimization metrics are {self._metrics}.")

    @final
    def _run_on_task(self, task: Dataset) -> None:
        if task is None:
            logger.error("Task run failed. Reason: task is undefined.")
            return

        if isinstance(task.X, np.ndarray) or isinstance(task.X, pd.DataFrame):
            preprocessor = DatasetPreprocessor()
            preprocessed_data = preprocessor.preprocess_data(task.X, task.y.squeeze())

            assert preprocessed_data is not None

            X, y = preprocessed_data
            X_train, X_test, y_train, y_test = preprocessor.split_data_on_train_and_test(X, y.squeeze())
        else:
            raise TypeError(f"pd.DataFrame or np.ndarray was expected. Got: {type(task.X)}")

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

        dataset_size_in_mb = int(pd.DataFrame(X_train).memory_usage(deep=True).sum() / (1024 ** 2))
        logger.info(f"Train sample size is {dataset_size_in_mb} mb.")
        if isinstance(self._automl, Imbaml):
            self._automl.dataset_size = dataset_size_in_mb

        for metric in self._metrics:
            start_time = time.time()
            self._automl.fit(X_train, y_train, metric, task.target_label)
            logger.info(f"Training on dataset (id={task.id}, name={task.name}) successfully finished.")

            time_passed = time.time() - start_time
            logger.info(f"Training time is {time_passed // 60} min.")

            y_predictions = self._automl.predict(X_test)
            # TODO: evaluate on additional metrics for a single runner.
            self._automl.score(metric, y_test, y_predictions, positive_class_label)


class AutoMLSingleRunner(AutoMLRunner):
    def __init__(self, task: Dataset, metric: str = 'f1', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = [metric]
        self._task = task

        self._configure_environment()

    @Decorators.log_exception
    def run(self) -> None:
        self._run_on_task(self._task)


class AutoMLBenchmarkRunner(AutoMLRunner):
    def __init__(self, metrics: Optional[List[str]], repository: DatasetRepository = ZenodoRepository(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        if metrics is None:
            self._metrics = ['f1']
        else:
            self._metrics = metrics
        self._repository = repository

        self._configure_environment()

    @property
    def repository(self):
        return self._repository

    @Decorators.log_exception
    def run(self) -> None:
        for task in self._repository.get_datasets():
            self._run_on_task(task)

