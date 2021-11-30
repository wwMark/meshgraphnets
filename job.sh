#!/bin/bash

sbatch --gres=gpu --time=2-00:00:00 noripple_sum_noattention.sh
sleep 2
sbatch --gres=gpu --time=2-00:00:00 noripple_min_noattention.sh
sleep 2
sbatch --gres=gpu --time=2-00:00:00 noripple_sum_attention.sh
sleep 2
sbatch --gres=gpu --time=2-00:00:00 ripple_sum_noattention.sh
sleep 2
sbatch --gres=gpu --time=2-00:00:00 ripple_sum_gradient_noattention.sh
