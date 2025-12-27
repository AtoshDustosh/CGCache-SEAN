#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

dataset="YouTube"
n_layers=2
n_epochs=3
bs=400

gpu=0

# # python ./train_self_supervised.py --data wikipedia --n_layers 2 --n_epochs 4 --gpu 7 --profile --profile_dir tracings/train_baseline
runtype="baseline"
profile_dir="tracings/${dataset}_${n_layers}_${runtype}"
python train_self_supervised.py --data "${dataset}" --bs "${bs}" \
    --n_layers "${n_layers}" --n_epochs "${n_epochs}" --gpu "${gpu}" \
    --profile --profile_dir "${profile_dir}"

# python ./train_self_supervised.py --data wikipedia --n_layers 2 --n_epochs 4 --gpu 7 --profile --profile_dir tracings/train_cache_0.05 --cg_cache 0.05
cache_rate=0.1
runtype="cache_${cache_rate}"
profile_dir="tracings/${dataset}_${n_layers}_${runtype}"
python train_self_supervised.py --data "${dataset}" --bs "${bs}" \
    --n_layers "${n_layers}" --n_epochs "${n_epochs}" --gpu "${gpu}" \
    --profile --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}"

# python ./train_self_supervised.py --data wikipedia --n_layers 2 --n_epochs 4 --gpu 7 --profile --profile_dir tracings/train_cache_0.05_redoNS --cg_cache 0.05 --redo_NS
cache_rate=0.1
runtype="cache_${cache_rate}_redoNS"
profile_dir="tracings/${dataset}_${n_layers}_${runtype}"
python train_self_supervised.py --data "${dataset}" --bs "${bs}" \
    --n_layers "${n_layers}" --n_epochs "${n_epochs}" --gpu "${gpu}" \
    --profile --profile_dir "${profile_dir}" \
    --cg_cache "${cache_rate}" \
    --redo_NS