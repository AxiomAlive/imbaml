#!/bin/bash

args=("$@")


run_experiment() {
  declare out='file'
  if [[ ${args[1]} == "c" ]]; then
    out="console"
  elif [[ ${args[1]} != "f" ]]; then
    echo "Invalid second argument. Options available: ['f' (for file), 'c' (for console)]."
    return
  fi

  declare automl='imba'
  if [[ ${args[0]} == "ag" || ${args[0]} == "imba" ]]; then  
    source devenv/bin/activate
    if [[ ${args[0]} == "ag" ]]; then
    	automl="ag"
    fi
    "$VIRTUAL_ENV"/bin/python -m experiment.main --automl="$automl" --o="$out"
  else
    echo "Invalid first argument. Options available: ['imba', 'ag']."
    return
  fi
}

run_experiment

