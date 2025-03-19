#!/bin/bash

run_experiment() {
  declare log_to_filesystem=true
  declare automl='imba'
  declare preset='best_quality'
  declare metric='f1'

  #TODO: add tasks argument.

  # TODO: check for flags for comparison.
  if [[ "$*" == *"console"* ]]; then
    # equals to false
    unset log_to_filesystem
  fi

  if [[ "$*" == *"ag"* ]]; then
    automl="ag"
  fi

  if [[ "$*" == *"flaml"* ]]; then
    automl="flaml"
  fi

  if [[ "$*" == *"acc"* ]]; then
      metric="balanced_accuracy"
    fi

  if [[ "$*" == *"ap"* ]]; then
        metric="average_precision"
      fi

  if [[ "$automl" == "imba" ]] || [[ ! -d ./devenv ]]; then
    source env/bin/activate
  else
    source devenv/bin/activate
  fi

  "$VIRTUAL_ENV"/bin/python -m experiment.main --automl="$automl" --log_to_filesystem="$log_to_filesystem" --preset="$preset" --metric="$metric"
}


run_experiment "$@"

