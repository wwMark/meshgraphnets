#!/bin/bash

sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_pna_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_min_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_attention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_ripple_sum_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_ripple_sum_gradient_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_ripple_sum_random_nodes_noattention.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_ripple_sum_density_noattention.sh
