#!/bin/bash

run_experiment() {
  declare out='file'
  declare automl='imba'
  declare preset='good_quality'
  declare trials=0
  #TODO: add tasks argument.

  # TODO: check for flags for comparison.
  if [[ "$*" == *"c"* ]]; then
    out="console"
  fi

  if [[ "$*" == *"ag"* ]]; then
    automl="ag"
  fi

  if [[ "$*" == *"50"* ]]; then
    trials=50
  fi

  if [[ "$*" == *"10"* ]]; then
      trials=10
    fi

  source env/bin/activate

  "$VIRTUAL_ENV"/bin/python -m experiment.main --automl="$automl" --out="$out" --preset="$preset" --trials="$trials"
}


run_experiment "$@"

