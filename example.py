import logging
import sys

from benchmark.repository import ZenodoRepository
from common.runner import AutoMLSingleRunner
from imblearn.datasets import fetch_datasets

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

    dataset = ZenodoRepository().load_dataset(1)

    runner = AutoMLSingleRunner(dataset, 'f1')
    runner.run()


if __name__ == '__main__':
    main()
