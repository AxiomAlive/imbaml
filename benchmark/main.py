import argparse
import logging
import sys
from datetime import datetime

from pathlib import Path
from common.runner import AutoMLBenchmarkRunner
import importlib

logger = logging.getLogger(__name__)


class AutoMLBenchmark:
    @staticmethod
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--automl', action='store', dest='automl', default='imbaml')
        parser.add_argument('--log_to_filesystem', action='store', dest='log_to_filesystem', type=bool, default=True)
        parser.add_argument('--autogluon_preset', action='store', dest='autogluon_preset', default='medium_quality')
        parser.add_argument('--metrics', action='store', dest='metric', default='f1')
        parser.add_argument('--sanity_check', action='store', dest='sanity_check', type=bool, default=False)

        args = parser.parse_args()
        automl = getattr(args, 'automl')
        log_to_filesystem = getattr(args, 'log_to_filesystem')
        ag_preset = getattr(args, 'autogluon_preset')
        metrics = getattr(args, 'metric').split(" ")
        sanity_check = getattr(args, 'sanity_check')

        for metric_name in metrics:
            if metric_name not in ['f1', 'balanced_accuracy', 'average_precision']:
                raise ValueError(
                    """
                    Invalid --metric option.
                    Options available: ['f1', 'balanced_accuracy', 'average_precision'].
                    """)

        logging_handlers = [
            logging.StreamHandler(stream=sys.stdout),
        ]

        if log_to_filesystem:
            log_filepath = 'logs/'
            if automl == 'ag':
                log_filepath += 'AutoGluon/'
            elif automl == 'imbaml':
                # TODO: rename dir.
                log_filepath += 'Imba/'
            elif automl == 'flaml':
                log_filepath += 'FLAML/'
            else:
                raise ValueError(
                    """
                    Invalid --automl option.
                    Options available: ['imbaml', 'ag', 'flaml'].
                    """)
                  
            Path(log_filepath).mkdir(parents=True, exist_ok=True)
            log_filepath += datetime.now().strftime('%Y-%m-%d %H:%M') + '.log'
            logging_handlers.append(logging.FileHandler(filename=log_filepath, encoding='utf-8', mode='w'))
        

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=logging_handlers
        )

        if automl == 'imbaml':
            bench_runner = AutoMLBenchmarkRunner(metrics, automl='imbaml', is_sanity_check=sanity_check)
        elif automl == 'ag':
            if ag_preset not in ['medium_quality', 'good_quality', 'high_quality', 'best_quality', 'extreme_quality']:
                raise ValueError(
                    """
                    Invalid --preset option.
                    Options available: ['medium_quality', 'good_quality', 'high_quality', 'best_quality', 'extreme_quality'].
                    """)
            bench_runner = AutoMLBenchmarkRunner(metrics, automl='ag', preset=ag_preset)
        elif automl == 'flaml':
            bench_runner = AutoMLBenchmarkRunner(metrics, automl='flaml')
        else:
            raise ValueError(
                """
                Invalid --automl option.
                Options available: ['imbaml', 'ag', 'flaml'].
                """)

        dataset_repo = bench_runner.repository
        dataset_repo.define_tasks()

        bench_runner.run()


if __name__ == '__main__':
    AutoMLBenchmark.main()

