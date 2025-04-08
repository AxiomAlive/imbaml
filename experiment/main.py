import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Optional

from setuptools import setup

import numpy as np
from pathlib import Path


class ExperimentMain:
    @staticmethod
    def run():
        parser = argparse.ArgumentParser()
        parser.add_argument('--automl', action='store', dest='automl', default='imba')
        parser.add_argument('--log_to_filesystem', action='store', dest='log_to_filesystem', type=bool, default=True)
        parser.add_argument('--preset', action='store', dest='preset', default=None)
        parser.add_argument('--trials', action='store', dest='trials', type=int, default=None)
        parser.add_argument('--metrics', action='store', dest='metric', default='f1')

        args = parser.parse_args()
        automl = getattr(args, 'automl')
        log_to_filesystem = getattr(args, 'log_to_filesystem')
        autogluon_preset = getattr(args, 'preset')
        trials = getattr(args, 'trials')
        metrics = getattr(args, 'metric').split(" ")

        if trials is not None and trials == 0:
            trials = None

        for metric_name in metrics:
            if metric_name not in ['f1', 'balanced_accuracy', 'average_precision', 'recall', 'precision']:
                raise ValueError(
                    """
                    Invalid --metric option.
                    Options available: ['f1', 'balanced_accuracy', 'average_precision', 'recall', 'precision'].
                    """)

        log_filepath = 'logs/'
        if automl == 'ag':
            log_filepath += 'AutoGluon/'
        elif automl == 'imba':
            log_filepath += 'Imba/'
        elif automl == 'flaml':
            log_filepath += 'FLAML/'
        else:
            raise ValueError(
                """
                Invalid --automl option.
                Options available: ['imba', 'ag', 'flaml'].
                """)

        logging_handlers = [
            logging.StreamHandler(stream=sys.stdout),
        ]
        if log_to_filesystem:
            Path(log_filepath).mkdir(parents=True, exist_ok=True)
            log_filepath += datetime.now().strftime('%Y-%m-%d %H:%M') + '.log'
            logging_handlers.append(logging.FileHandler(filename=log_filepath, encoding='utf-8', mode='w'))

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=logging_handlers
        )

        if automl == 'ag':
            if autogluon_preset not in ['medium_quality', 'good_quality', 'high_quality', 'best_quality', None]:
                raise ValueError(
                    """
                    Invalid --preset option.
                    Options available: ['medium_quality', 'good_quality', 'high_quality', 'best_quality'].
                    """)

            from experiment.autogluon import AutoGluonExperimentRunner
            automl_runner = AutoGluonExperimentRunner(preset=autogluon_preset, metrics=metrics)
        elif automl == 'imba':
            from experiment.imba import ImbaExperimentRunner
            automl_runner = ImbaExperimentRunner(metrics)
        elif automl == 'flaml':
            from experiment.flaml_automl import FLAMLExperimentRunner
            automl_runner = FLAMLExperimentRunner(metrics)
        else:
            raise ValueError(
                """
                Invalid --automl option.
                Options available: ['imba', 'ag', 'flaml'].
                """)

        benchmark_runner = automl_runner.benchmark_runner
        benchmark_runner.define_tasks()

        automl_runner.run(trials)


if __name__ == '__main__':
    ExperimentMain.run()

