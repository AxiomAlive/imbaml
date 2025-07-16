import logging
from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score

from experiment.runner import AutoMLExperimentRunner
from flaml import AutoML


logger = logging.getLogger(__name__)


class FLAMLRunner(AutoMLExperimentRunner):
    def __init__(self, metrics):
        super().__init__(metrics)

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        metric_name: str,
        target_label: Optional[str],
        dataset_name: str
    ) -> None:

        if metric_name == 'average_precision':
            self._metric_automl_arg = 'ap'
        elif metric_name == 'f1':
            self._metric_automl_arg = 'f1'
        elif metric_name in ['balanced_accuracy', 'precision', 'recall']:
            raise ValueError(f"Metric {metric_name} is not supported.")

        automl = AutoML()
        automl.fit(X_train, y_train, task='classification', metric=self._metric_automl_arg)

        best_loss = automl.best_loss
        best_model = automl.best_estimator
        self._log_val_loss_alongside_model_class({best_model: best_loss})

        self._fitted_model = automl
