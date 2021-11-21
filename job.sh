#!/bin/bash

sbatch --gres=gpu noripple_sum_noattention.sh
sbatch --gres=gpu noripple_min_noattention.sh
sbatch --gres=gpu noripple_sum_attention.sh
sbatch --gres=gpu ripple_sum_noattention.sh
