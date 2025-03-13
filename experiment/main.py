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
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--automl', action='store', dest='automl', default='imba')
        parser.add_argument('--out', action='store', dest='out', default='file')
        parser.add_argument('--preset', action='store', dest='preset', default='good_quality')
        parser.add_argument('--trials', action='store', dest='trials', type=int, default=None)
        parser.add_argument('--tasks', action='store', dest='tasks', type=tuple, nargs='*', default=None)
        parser.add_argument('--metric', action='store', dest='metric', default='f1')

        args = parser.parse_args()
        automl = getattr(args, 'automl')
        logging_output = getattr(args, 'out')
        autogluon_preset = getattr(args, 'preset')
        trials = getattr(args, 'trials')
        tasks = getattr(args, 'tasks')
        metric_name = getattr(args, 'metric')

        if trials is not None and trials == 0:
            trials = None

        if metric_name not in ['f1', 'balanced_accuracy', 'average_precision']:
            raise ValueError("Invalid --metric option. Options available: ['f1', 'balanced_accuracy', 'average_precision'].")

        if automl == 'ag':
            if autogluon_preset not in ['medium_quality', 'good_quality', 'high_quality', 'best_quality']:
                raise ValueError("Invalid --preset option. Options available: ['medium_quality', 'good_quality', 'high_quality', 'best_quality'].")

            from experiment.autogluon import AutoGluonExperimentRunner

            automl_runner = AutoGluonExperimentRunner(preset=autogluon_preset, metric=metric_name)
        elif automl == 'imba':
            from experiment.imba import ImbaExperimentRunner

            automl_runner = ImbaExperimentRunner(metric_name)
        elif automl == 'flaml':
            from experiment.flaml_automl import FLAMLExperimentRunner

            automl_runner = FLAMLExperimentRunner(metric_name)
        else:
            raise ValueError("Invalid --automl option. Options available: ['imba', 'ag'].")

        if logging_output == 'file':
            if automl == 'imba':
                log_file_name = 'logs/Imba/'
            elif automl == 'ag':
                log_file_name = 'logs/AG/'
            elif automl == 'flaml':
                log_file_name = 'logs/FLAML/'

            Path(log_file_name).mkdir(parents=True, exist_ok=True)
            log_file_name += datetime.now().strftime('%Y-%m-%d %H:%M') + '.log'

            logging.basicConfig(
                filename=log_file_name,
                filemode="a",
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        elif logging_output == 'console':
            logging.basicConfig(
                stream=sys.stdout,
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        else:
            raise ValueError("Invalid --out option. Options available: ['file', 'console'].")

        benchmark_runner = automl_runner.get_benchmark_runner()
        benchmark_runner.define_tasks(tasks)

        automl_runner.run(trials)




if __name__ == '__main__':
    ExperimentMain.main()

