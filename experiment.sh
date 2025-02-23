#!/bin/bash

run_experiment() {
  declare out='file'
  declare automl='imba'
  declare preset='good_quality'
  #TODO: add tasks argument.

  # TODO: check for flags for comparison.
  if [[ "$*" == *"c"* ]]; then
    out="console"
  fi

  if [[ "$*" == *"ag"* ]]; then
    automl="ag"
  fi

  if [[ "$*" == *"flaml"* ]]; then
    automl="flaml"
  fi

  if [[ "$automl" == "imba" ]]; then
    source env/bin/activate
  else
    source devenv/bin/activate
  fi

  "$VIRTUAL_ENV"/bin/python -m experiment.main --automl="$automl" --out="$out" --preset="$preset"
}


run_experiment "$@"

