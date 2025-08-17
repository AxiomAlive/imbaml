#!/bin/bash

prepare_environment() {
  declare -g log_to_filesystem=true
  # TODO: check for flags for comparison.
  if [[ "$*" == *"-c"* ]]; then
    unset log_to_filesystem
  fi

  declare -g automl='imbaml'
  #  declare imba_search_space='all'
  case "$*" in
    "-ag"*)
      automl="ag"
      declare -g autogluon_preset='medium_quality'

      case "$*" in
        *"-ag_good"*)
          autogluon_preset='good_quality'
        ;;
        *"-ag_best"*)
          autogluon_preset='best_quality'
        ;;
        *"-ag_extreme"*)
          autogluon_preset='extreme_quality'
        ;;
      esac
    ;;
    *"-flaml"*)
      automl="flaml"
      ;;
  esac

  declare -g metrics=('f1' 'balanced_accuracy' 'average_precision')
  case "$*" in
    *"-f1"*)
      metrics=("f1")
      ;;
    *"-bal_acc"*)
      metrics=("balanced_accuracy")
      ;;
    *"-ap"*)
      metrics=("average_precision")
      ;;
  esac

  declare -g sanity_check=true
  if [[ "$*" != *"-test"* || $automl != 'imbaml' ]]; then
    unset sanity_check
  fi

  source venv/bin/activate
}

run_on_cloud() {
  prepare_environment "$@"

  datasphere project job execute -p "$YC_PROJECT_ID" -c cloud.yaml &
}

run_locally() {
  prepare_environment "$@"

  "$VIRTUAL_ENV"/bin/python -m benchmark.main\
  --automl="$automl"\
  --log_to_filesystem="$log_to_filesystem"\
  --autogluon_preset="$autogluon_preset"\
  --metrics="${metrics[*]}"\
  --sanity_check="$sanity_check"
}

if [[ "$*" == *"-cloud"* ]]; then
  run_on_cloud "$@"
else
  run_locally "$@"
fi


