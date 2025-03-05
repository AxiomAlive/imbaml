import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score

from experiment.runner import AutoMLRunner
from flaml import AutoML


logger = logging.getLogger(__name__)


class FLAMLExperimentRunner(AutoMLRunner):
    def __init__(self, metric):
        super().__init__(metric)

    def fit(self,
            X_train: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            target_label: str,
            dataset_name: str,
            n_evals: int) -> None:
        automl = AutoML()
        automl.fit(X_train, y_train, task='classification', time_budget=-1, metric=self._metric)

        best_loss = automl.best_loss
        best_model = automl.best_estimator
        self._log_val_loss_alongside_model_class({best_model: best_loss})

        self._fitted_model = automl
