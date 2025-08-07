import logging
import sys
from collections import Counter

import numpy as np

from common.preprocessing import DatasetPreprocessor
from imblearn.datasets import fetch_datasets

from benchmark.imbaml_ import Imbaml

logger = logging.getLogger(__name__)


def main():
    logging_handlers = [
        logging.StreamHandler(stream=sys.stdout),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=logging_handlers
    )

    dataset = fetch_datasets()['ecoli']

    preprocessor = DatasetPreprocessor()
    preprocessed_data = preprocessor.preprocess_data(dataset.get('data'), dataset.get('target').squeeze())

    X, y = preprocessed_data
    X_train, X_test, y_train, y_test = preprocessor.split_data_on_train_and_test(X, y.squeeze())

    class_belongings = Counter(y_train)

    iterator_of_class_belongings = iter(sorted(class_belongings))
    *_, positive_class_label = iterator_of_class_belongings

    runner = Imbaml(is_sanity_check=True)
    runner.fit(X_train, y_train, 'f1', None, 'ecoli')

    y_predictions = runner.predict(X_test)
    runner.score('f1', y_test, y_predictions, positive_class_label)


if __name__ == '__main__':
    main()
