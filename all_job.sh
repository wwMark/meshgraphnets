#!/bin/bash

sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_attention_15.sh
sleep 2

######
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_noattention_3.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_noattention_3.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_noattention_3.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_noattention_3.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_sum_noattention_3.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_sum_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_sum_noattention_15.sh
sleep 2

###
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_attention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_attention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_attention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_attention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_attention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_attention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_attention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_attention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_attention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_attention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_attention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_attention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_attention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_attention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_attention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_attention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_attention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_attention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_attention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_attention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_sum_attention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_sum_attention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_sum_attention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_sum_attention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_sum_attention_15.sh
sleep 2