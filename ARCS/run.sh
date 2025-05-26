#!/bin/bash

START_CORE=0
NUM_CORES=12
SEEDS=(1 2 3 4 5)
# ENV_LIST = ['multicomp/SumoHumans-v0', 'multicomp/YouShallNotPassHumans-v0', 'multicomp/KickAndDefend-v0']

for IDX in "${!SEEDS[@]}"
do
  SEED=${SEEDS[$IDX]}
  START=$((START_CORE + IDX * NUM_CORES))
  END=$((START + NUM_CORES - 1))
  CORES="${START}-${END}"

  LOG_FILE="sumo_llm${SEED}.log"
  CMD="nohup taskset -c ${CORES} python -u ARCS/src/my_entrance.py --seed $SEED --mode 'llm' --env 'multicomp/SumoHumans-v0' > $LOG_FILE 2>&1 &"

  echo "Launching seed $SEED on cores $CORES: $CMD"
  eval $CMD
  echo "Started seed $SEED with PID $!"
done
