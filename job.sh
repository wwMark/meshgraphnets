#!/bin/bash

sbatch --gres=gpu -t 48:00:00 -p gpu_8 noripple_sum_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 noripple_min_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 noripple_sum_attention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 ripple_sum_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 ripple_sum_gradient_noattention.sh
