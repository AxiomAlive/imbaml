#!/bin/bash

args=("$@")


run_experiment() {
  declare out='file'
  declare automl='imba'
  declare preset='good_quality'
  declare trials=30

  # TODO: check for flags for comparison.
  if [[ ${args[1]} == "c" ]]; then
    out="console"
  elif [[ ${args[1]} != "f" ]]; then
    echo "Invalid second argument. Options available: ['f' (for file), 'c' (for console)]."
    return
  fi
  declare
  if [[ ${args[0]} == "ag" || ${args[0]} == "imba" ]]; then
    source env/bin/activate
    if [[ ${args[0]} == "ag" ]]; then
    	automl="ag"
    fi
    "$VIRTUAL_ENV"/bin/python -m experiment.main --automl="$automl" --out="$out" --preset="$preset" --trials="$trials"
  else
    echo "Invalid first argument. Options available: ['imba', 'ag']."
    return
  fi
}

run_experiment

