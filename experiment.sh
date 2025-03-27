#!/bin/bash

run_experiment() {
  declare log_to_filesystem=true
  declare automl='imba'
  declare preset='best_quality'
  declare metrics=('f1')

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
    metric[0]="balanced_accuracy"
  fi

  if [[ "$*" == *"rec"* ]]; then
      metric[0]="recall"
  fi

  if [[ "$*" == *"pr"* ]]; then
      metric[0]="precision"
  fi

  if [[ "$*" == *"ap"* ]]; then
      metric[0]="average_precision"
  fi

   if [[ "$*" == *"pr"* && "$*" == *"rec"* ]]; then
        metric[0]="precision"
        metric[1]="recall"
    fi

  if [[ "$automl" == "imba" ]] || [[ ! -d ./devenv ]]; then
    source env/bin/activate
  else
    source devenv/bin/activate
  fi

  "$VIRTUAL_ENV"/bin/python -m experiment.main --automl="$automl" --log_to_filesystem="$log_to_filesystem" --preset="$preset" --metrics="${metrics[*]}"
}


run_experiment "$@"

