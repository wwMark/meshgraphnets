#!/bin/bash

sbatch --gres=gpu noripple_sum_noattention.sh
sleep 2
sbatch --gres=gpu noripple_min_noattention.sh
sleep 2
sbatch --gres=gpu noripple_sum_attention.sh
sleep 2
sbatch --gres=gpu ripple_sum_noattention.sh
sleep 2
sbatch --gres=gpu ripple_sum_gradient_noattention.sh
