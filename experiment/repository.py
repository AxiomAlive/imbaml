import itertools
import logging
import multiprocessing
import os
import pprint
import shutil
import traceback
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from typing import Tuple, Optional, Union, List, Callable, Any, TypeVar

import time
import numpy as np
import pandas as pd
import sklearn.base
from imblearn.datasets import fetch_datasets
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import *

from domain import Dataset
from utils.decorators import ExceptionWrapper

logger = logging.getLogger(__name__)
FittedModel = TypeVar('FittedModel', bound=Any)


class BenchmarkExperimentRunner(ABC):
    def __init__(self, *args, **kwargs):
        self._tasks: List[Dataset, ...] = []
        self._id_counter = itertools.count(start=1)

    @abstractmethod
    def define_tasks(self, task_range: Optional[List[int]] = None):
        raise NotImplementedError()

    @abstractmethod
    def load_dataset(self, task_id: Optional[int] = None) -> Optional[Dataset]:
        raise NotImplementedError()

    def get_tasks(self):
        return self._tasks


class ZenodoExperimentRunner(BenchmarkExperimentRunner):
    def __init__(self):
        super().__init__()
        self._datasets = fetch_datasets(data_home='datasets/imbalanced', verbose=True)

        os.environ['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'
        os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'
        os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '1'

    def load_dataset(self, task_id: Optional[int] = None) -> Optional[Dataset]:
        for i, (dataset_name, dataset_data) in enumerate(self._datasets.items()):
            if i + 1 == task_id:
                return Dataset(
                    id=next(self._id_counter),
                    name=dataset_name,
                    X=dataset_data.get('data'),
                    y=dataset_data.get('target'))

    def define_tasks(self, task_range: Optional[List[int]] = None):
        if task_range is None:
            task_range = range(1, len(self._datasets.keys()) + 1)
            logger.info(task_range)
        for i in task_range:
            self._tasks.append(self.load_dataset(i))

    def fit(
            self,
            X_train: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            target_label: str,
            dataset_name: str):
        raise NotImplementedError

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        raise NotImplementedError


class OpenMLExperimentRunner(BenchmarkExperimentRunner):
    def __init__(self):
        super().__init__()

        import openml
        openml.config.set_root_cache_directory("openml_cache")

    def load_dataset(self, task_id: Optional[int] = None) -> Optional[Dataset]:
        try:
            with multiprocessing.Pool(processes=1) as pool:
                task = pool.apply_async(openml.tasks.get_task, [task_id]).get(timeout=1800)
                dataset = pool.apply_async(task.get_dataset, []).get(timeout=1800)
            X, y, categorical_indicator, dataset_feature_names = dataset.get_data(
                target=dataset.default_target_attribute)

        except multiprocessing.TimeoutError:
            logger.error(f"Fetch from OpenML timed out. Dataset id={task_id} was not loaded.")
            return None
        except Exception as exc:
            logger.error(pprint.pformat(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            return None

        return Dataset(
            id=next(self._id_counter),
            name=dataset.name,
            target_label=dataset.default_target_attribute,
            X=X,
            y=y)

    def define_tasks(self, task_range: List[int] = None):
        self._tasks = []
        benchmark_suite = openml.study.get_suite(suite_id=271)

        for i, task_id in enumerate(benchmark_suite.tasks):
            if task_range is not None and i not in task_range:
                continue

            self._tasks.append(self.load_dataset(task_id))
