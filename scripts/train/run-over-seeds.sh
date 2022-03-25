#!/bin/bash

SEEDS="0 1 2 3 4"
EXEC=$@

for seed in $SEEDS; do
    echo "RANDOM SEED $seed";
    $EXEC --seed $seed;
done
