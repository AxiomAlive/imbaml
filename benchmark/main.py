import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
from setuptools import setup

import numpy as np
from pathlib import Path


class ApplicationMain:
    @staticmethod
    def run():
        parser = argparse.ArgumentParser()
        parser.add_argument('--automl', action='store', dest='automl', default='imba')
        parser.add_argument('--log_to_filesystem', action='store', dest='log_to_filesystem', type=bool, default=True)
        parser.add_argument('--autogluon_preset', action='store', dest='autogluon_preset', default='good_quality')
        parser.add_argument('--metrics', action='store', dest='metric', default='f1')
        parser.add_argument('--sanity_check', action='store', dest='sanity_check', type=bool, default=False)

        args = parser.parse_args()
        automl = getattr(args, 'automl')
        log_to_filesystem = getattr(args, 'log_to_filesystem')
        ag_preset = getattr(args, 'autogluon_preset')
        metrics = getattr(args, 'metric').split(" ")
        sanity_check = getattr(args, 'sanity_check')

        for metric_name in metrics:
            if metric_name not in ['f1', 'balanced_accuracy', 'average_precision', 'recall', 'precision']:
                raise ValueError(
                    """
                    Invalid --metric option.
                    Options available: ['f1', 'balanced_accuracy', 'average_precision', 'recall', 'precision'].
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
        
        import importlib
        importlib.reload(logging)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=logging_handlers
        )

        if automl == 'imbaml':
            from benchmark.imbaml_ import ImbamlRunner
            automl_runner = ImbamlRunner(metrics, is_sanity_check=sanity_check)
        elif automl == 'ag':
            if ag_preset not in ['medium_quality', 'good_quality', 'high_quality', 'best_quality']:
                raise ValueError(
                    """
                    Invalid --preset option.
                    Options available: ['medium_quality', 'good_quality', 'high_quality', 'best_quality'].
                    """)

            from benchmark.autogluon import AutoGluonRunner
            automl_runner = AutoGluonRunner(preset=ag_preset, metrics=metrics)
        elif automl == 'flaml':
            from benchmark.flaml_ import FLAMLRunner
            automl_runner = FLAMLRunner(metrics)
        else:
            raise ValueError(
                """
                Invalid --automl option.
                Options available: ['imbaml', 'ag', 'flaml'].
                """)

        benchmark_runner = automl_runner.benchmark_runner
        benchmark_runner.define_tasks()

        automl_runner.run()


if __name__ == '__main__':
    ApplicationMain.run()

