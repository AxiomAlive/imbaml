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

        args = parser.parse_args()
        automl = getattr(args, 'automl')
        logging_output = getattr(args, 'out')
        autogluon_preset = getattr(args, 'preset')
        trials = getattr(args, 'trials')
        tasks = getattr(args, 'tasks')

        if trials is not None and trials == 0:
            trials = None

        if logging_output == 'file':
            if automl == 'ag':
                log_file_name = 'logs/AG/'
            else:
                log_file_name = 'logs/Imba/'

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

        if automl == 'ag':
            if autogluon_preset not in ['medium_quality', 'good_quality']:
                raise ValueError("Invalid --preset option. Options available: ['medium_quality', 'good_quality'].")

            from experiment.autogluon import AGExperimentRunner

            runner = AGExperimentRunner(autogluon_preset)
        elif automl == 'imba':
            from experiment.imba import ImbaExperimentRunner

            runner = ImbaExperimentRunner()
        else:
            raise ValueError("Invalid --automl option. Options available: ['imba', 'ag'].")

        runner.get_benchmark_runner().define_tasks(tasks)

        runner.run(trials)




if __name__ == '__main__':
    ExperimentMain.main()

