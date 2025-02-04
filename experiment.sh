#!/bin/bash

#args=("$@")
#n_args="$#"

run_experiment() {
  declare out='file'
  declare automl='imba'
  declare preset='good_quality'
  declare trials=30

  # TODO: check for flags for comparison.
  if [[ "$*" == *"c"* ]]
  then
    out="console"
  fi

  if [[ "$*" == *"c"* ]]
    then
      out="console"
    fi

  if [[ "$*" == *"50"* ]]
      then
        trials=50
      fi

#    if [[ ${args[2]} == "c" ]]; then
#        out="console"
#    fi
  #
  #  if [[ ${args[1]} == "c" ]]; then
  #      out="console"
  #    fi


  #  if [[ ${args[0]} == "ag" || ${args[0]} == "imba" ]]; then
  source env/bin/activate
  #  if [[ ${args[0]} == "ag" ]]; then
  #    automl="ag"
  #  fi
  "$VIRTUAL_ENV"/bin/python -m experiment.main --automl="$automl" --out="$out" --preset="$preset" --trials="$trials"
  #  else
  #    echo "Invalid first argument. Options available: ['imba', 'ag']."
  #    return
  #  fi
}


run_experiment "$@"

