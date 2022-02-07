#!/bin/bash
#SBATCH --partition=gpu_8

last_run="$1"
run_prefix="/pfs/data5/home/kit/anthropomatik/sn2444/meshgraphnets/output/deforming_plate/"
model_last_run_dir="$run_prefix$last_run"
echo "$model_last_run_dir"
srun --exclusive -N1 -p gpu_8 --gres=gpu python run_model.py --model_last_run_dir=${model_last_run_dir} --use_prev_config=True
