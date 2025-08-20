import logging
import sys

from benchmark.repository import ZenodoRepository
from common.runner import AutoMLSingleRunner
from imblearn.datasets import fetch_datasets

logger = logging.getLogger(__name__)


def main():
    dataset = ZenodoRepository().load_dataset(1)

    # runner = AutoMLSingleRunner(dataset, 'f1', sanity_check=True, verbosity=1)
    runner = AutoMLSingleRunner(dataset, 'f1')

    runner.run()


if __name__ == '__main__':
    main()
