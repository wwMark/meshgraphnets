#!/bin/bash

sbatch --gres=gpu -t 48:00:00 -p gpu_4 deform_noripple_sum_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_4 deform_noripple_min_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_4 deform_noripple_sum_attention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_4 deform_ripple_sum_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_4 deform_ripple_sum_gradient_noattention.sh
