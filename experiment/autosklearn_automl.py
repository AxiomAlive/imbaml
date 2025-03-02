import logging
from typing import Union

import numpy as np
import pandas as pd

from experiment.runner import AutoMLRunner
import autosklearn

logger = logging.getLogger(__name__)


class AutoSklearnExperimentRunner(AutoMLRunner):
    def fit(self,
            X_train: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            target_label: str,
            dataset_name: str,
            n_evals: int
            ) -> None:
        automl = autosklearn.classification.AutoSklearnClassifier()
        automl.fit(X_train, y_train, time_left_for_this_task=7200, metric=autosklearn.metrics.f1)

        run_history = automl.automl_.runhistory_

        best_run_id = min(run_history.data, key=lambda x: run_history.data[x]['cost'])
        best_loss = best_run_id.data[best_run_id]['cost']

        logger.info(automl.show_models())

        self._log_val_loss_alongside_model_class({'123': best_loss})

        self._fitted_model = automl
