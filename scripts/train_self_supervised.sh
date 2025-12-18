#!/bin/bash



python train_self_supervised.py --data wikipedia --seed 0 --gpu 1 --n_layers 2 \
    --n_epochs 5 --profile