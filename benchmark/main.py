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
        parser.add_argument('--metrics', action='store', dest='metrics', default='f1')
        parser.add_argument('--sanity_check', action='store', dest='sanity_check', type=bool, default=False)

        args = parser.parse_args()
        automl = getattr(args, 'automl')
        log_to_filesystem = getattr(args, 'log_to_filesystem')
        ag_preset = getattr(args, 'autogluon_preset')
        metrics = getattr(args, 'metrics').split(" ")
        sanity_check = getattr(args, 'sanity_check')

        for metric_name in metrics:
            if metric_name not in ['f1', 'balanced_accuracy', 'average_precision']:
                raise ValueError(
                    """
                    Invalid --metric option.
                    Options available: ['f1', 'balanced_accuracy', 'average_precision'].
                    """)

        if automl == 'imbaml':
            bench_runner = AutoMLBenchmarkRunner(
                metrics,
                automl='imbaml',
                sanity_check=sanity_check,
                log_to_file=log_to_filesystem
            )
        elif automl == 'ag':
            if ag_preset not in ['medium_quality', 'good_quality', 'high_quality', 'best_quality', 'extreme_quality']:
                raise ValueError(
                    """
                    Invalid --preset option.
                    Options available: ['medium_quality', 'good_quality', 'high_quality', 'best_quality', 'extreme_quality'].
                    """)
            bench_runner = AutoMLBenchmarkRunner(metrics, automl='ag', preset=ag_preset, log_to_filesystem=log_to_filesystem)
        elif automl == 'flaml':
            bench_runner = AutoMLBenchmarkRunner(metrics, automl='flaml', log_to_filesystem=log_to_filesystem)
        else:
            raise ValueError(
                """
                Invalid --automl option.
                Options available: ['imbaml', 'ag', 'flaml'].
                """)

        dataset_repo = bench_runner.repository
        dataset_repo.load_datasets()

        bench_runner.run()


if __name__ == '__main__':
    AutoMLBenchmark.main()

