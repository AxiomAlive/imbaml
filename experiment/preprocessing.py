import logging
from typing import Tuple, Union, Optional, List

import numpy as np
import pandas as pd
from imblearn.datasets import make_imbalance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts

logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    def preprocess_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]:
        if isinstance(X, pd.DataFrame):
            X.dropna(inplace=True)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        if isinstance(y, pd.Series):
            y_encoded = pd.Series(y_encoded)

        if isinstance(X, pd.DataFrame):
            X_encoded = X.copy(deep=True)
            for dataset_feature_name in X:
                dataset_feature = X.get(dataset_feature_name)

                if type(dataset_feature.iloc[0]) is str:
                    dataset_feature_encoded = pd.get_dummies(dataset_feature, prefix=dataset_feature_name)
                    X_encoded.drop([dataset_feature_name], axis=1, inplace=True)
                    X_encoded = pd.concat([X_encoded, dataset_feature_encoded], axis=1)
                    X_encoded.reset_index(drop=True, inplace=True)

            assert len(X_encoded.index) == len(y_encoded.index), f"X index size is {len(X_encoded.index)} and y index size is {len(y_encoded.index)}."
        else:
            X_encoded = X


        return X_encoded, y_encoded

    def split_data_on_train_and_test(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> List[Union[pd.DataFrame, pd.Series, np.ndarray]]:
        return tts(
            X,
            y,
            random_state=42,
            test_size=0.2,
            stratify=y)

    def make_imbalance(
        self,
        X_train,
        y_train,
        class_belongings,
        pos_label
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
        is_dataset_initially_imbalanced = True
        number_of_positives = class_belongings.get(pos_label)
        proportion_of_positives = number_of_positives / len(y_train)

        # For extreme case - 0.01, for moderate - 0.2, for mild - 0.4.
        if proportion_of_positives > 0.01:
            coefficient = 0.01
            updated_number_of_positives = int(coefficient * len(y_train))

            assert updated_number_of_positives < 10, "Number of positive class instances is too low."
            class_belongings[pos_label] = updated_number_of_positives
            is_dataset_initially_imbalanced = False

        if not is_dataset_initially_imbalanced:
            X_train, y_train = make_imbalance(
                X_train,
                y_train,
                sampling_strategy=class_belongings)
            logger.info("Imbalancing applied.")

        return X_train, y_train
