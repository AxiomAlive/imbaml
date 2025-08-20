import itertools
import logging
import multiprocessing
import os
import pprint
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any, TypeVar

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets

from common.domain import Dataset

logger = logging.getLogger(__name__)
FittedModel = TypeVar('FittedModel', bound=Any)


class DatasetRepository(ABC):
    def __init__(self, *args, **kwargs):
        self._datasets: List[Dataset, ...] = []
        self._id = itertools.count(start=1)

    @abstractmethod
    def load_datasets(self, task_range: Optional[List[int]] = None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load_dataset(self, task_id: Optional[int] = None) -> Dataset:
        raise NotImplementedError()

    def get_datasets(self):
        return self._datasets


class ZenodoRepository(DatasetRepository):
    def __init__(self):
        super().__init__()
        self._raw_datasets = fetch_datasets(data_home='tasks/imbalanced-learning', verbose=True)

    def load_dataset(self, task_id: Optional[int] = None) -> Dataset:
        for i, (dataset_name, dataset_data) in enumerate(self._raw_datasets.items(), 1):
            if i == task_id:
                return Dataset(
                    id=next(self._id),
                    name=dataset_name,
                    X=dataset_data.get('data'),
                    y=dataset_data.get('target'))
            elif i > task_id:
                raise ValueError(f"Dataset with id={task_id} is not available.")

    def load_datasets(self, task_range: Optional[List[int]] = None) -> None:
        if task_range is None:
            range_start = 1
            range_end = len(self._raw_datasets.keys()) + 1
            task_range = range(range_start, range_end)
            logger.info(f"Running tasks from {range_start} to {range_end}.")
        for i in task_range:
            self._datasets.append(self.load_dataset(i))


# TODO: refactor.
class OpenMLRepository(DatasetRepository):
    def __init__(self, suite_id=271):
        super().__init__()
        self._suit_id = suite_id

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
            id=next(self._id),
            name=dataset.name,
            target_label=dataset.default_target_attribute,
            X=X,
            y=y)

    def load_datasets(self, task_range: List[int] = None) -> None:
        benchmark_suite = openml.study.get_suite(suite_id=self._suit_id)

        for i, task_id in enumerate(benchmark_suite.tasks):
            if task_range is not None and i not in task_range:
                continue

            self._datasets.append(self.load_dataset(task_id))
