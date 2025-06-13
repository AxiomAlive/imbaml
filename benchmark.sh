#!/bin/bash

run_experiment() {
  declare log_to_filesystem=true
  declare automl='imba'
  declare autogluon_preset='good_quality'
  declare metrics=('f1')
  declare sanity_check=true

  # TODO: check for flags for comparison.
  if [[ "$*" == *"-c"* ]]; then
    unset log_to_filesystem
  fi

  if [[ "$*" == *"-ag"* ]]; then
    automl="ag"
  else
    unset autogluon_preset
  fi

  if [[ "$*" == *"-flaml"* ]]; then
    automl="flaml"
  fi

  if [[ "$*" == *"-acc"* ]]; then
    metrics[0]="balanced_accuracy"
  fi

  if [[ "$*" == *"-rec"* ]]; then
      metrics[0]="recall"
  fi

  if [[ "$*" == *"-pr"* ]]; then
      metrics[0]="precision"
  fi

  if [[ "$*" == *"-ap"* ]]; then
      metrics[0]="average_precision"
  fi

   if [[ "$*" == *"-pr"* && "$*" == *"-rec"* ]]; then
      metrics[0]="precision"
      metrics[1]="recall"
  fi

  if [[ "$*" != *"-test"* ]]; then
    unset sanity_check
  fi

  if [[ "$automl" == "imba" ]] || [[ ! -d ./devenv ]]; then
    source env/bin/activate
  else
    source devenv/bin/activate
  fi


  "$VIRTUAL_ENV"/bin/python -m experiment.main\
  --automl="$automl"\
  --log_to_filesystem="$log_to_filesystem"\
  --autogluon_preset="$autogluon_preset"\
  --metrics="${metrics[*]}"\
  --sanity_check="$sanity_check"
}


run_experiment "$@"

