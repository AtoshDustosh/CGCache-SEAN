#!/bin/bash

export CUDA_VISIBLE_DEVICES=6

dataset="reddit"
n_layers=2
n_epochs=5
bs=200
gpu=0

model="SEAN"

# ================= baseline =================
runtype="baseline"
profile_dir="tracings/${model}_${dataset}_${n_layers}_${runtype}"

python train_self_supervised.py --data "${dataset}" --bs "${bs}" \
    --n_layers "${n_layers}" --n_epochs "${n_epochs}" --gpu "${gpu}" \
    --profile --profile_dir "${profile_dir}"

# ================= cache experiments =================
cache_rates=(0.05 \
            0.1 0.2 0.5 1.0 \
            )

for cache_rate in "${cache_rates[@]}"; do
    # ---- normal cache ----
    runtype="cache_${cache_rate}"
    profile_dir="tracings/${model}_${dataset}_${n_layers}_${runtype}"

    python train_self_supervised.py --data "${dataset}" --bs "${bs}" \
        --n_layers "${n_layers}" --n_epochs "${n_epochs}" --gpu "${gpu}" \
        --profile --profile_dir "${profile_dir}" \
        --cg_cache "${cache_rate}"

    # ---- cache + redo_NS ----
    runtype="cache_${cache_rate}_redoNS"
    profile_dir="tracings/${model}_${dataset}_${n_layers}_${runtype}"

    python train_self_supervised.py --data "${dataset}" --bs "${bs}" \
        --n_layers "${n_layers}" --n_epochs "${n_epochs}" --gpu "${gpu}" \
        --profile --profile_dir "${profile_dir}" \
        --cg_cache "${cache_rate}" \
        --redo_NS

    # ---- cache + redo_NS + async_cache----
    runtype="cache_${cache_rate}_redoNS_async"
    profile_dir="tracings/${model}_${dataset}_${n_layers}_${runtype}"

    python train_self_supervised.py --data "${dataset}" --bs "${bs}" \
        --n_layers "${n_layers}" --n_epochs "${n_epochs}" --gpu "${gpu}" \
        --cg_cache "${cache_rate}" \
        --profile --profile_dir "${profile_dir}" \
        --redo_NS --async_cache
done
