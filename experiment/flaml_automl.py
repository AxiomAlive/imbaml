from typing import Union

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score

from experiment.runner import AutoMLRunner
from flaml import AutoML


class FLAMLExperimentRunner(AutoMLRunner):
    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series], target_label: str,
            dataset_name: str) -> None:
        flaml = AutoML()
        flaml.fit(X_train, y_train, task='classification', time_budget=-1, metric='f1')

        best_loss = flaml.best_loss
        best_model = flaml.best_estimator
        self._log_val_loss_alongside_model_class({best_model: best_loss})

        self._fitted_model = flaml

