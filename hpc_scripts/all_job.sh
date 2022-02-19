#!/bin/bash

sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_max_noattention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_max_noattention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_max_noattention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_max_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_max_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_mean_noattention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_mean_noattention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_mean_noattention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_mean_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_mean_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_min_noattention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_min_noattention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_min_noattention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_min_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_min_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_pna_noattention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_pna_noattention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_pna_noattention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_pna_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_pna_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_noattention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_noattention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_noattention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_noattention_15.sh
sleep 2

###
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_max_attention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_max_attention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_max_attention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_max_attention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_max_attention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_mean_attention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_mean_attention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_mean_attention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_mean_attention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_mean_attention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_min_attention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_min_attention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_min_attention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_min_attention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_min_attention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_pna_attention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_pna_attention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_pna_attention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_pna_attention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_pna_attention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_attention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_attention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_attention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_attention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 deform_noripple_sum_attention_15.sh
sleep 2

###
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_distancedensity_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_distancedensity_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_distancedensity_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_distancedensity_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_distancedensity_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_equalsize_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_equalsize_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_equalsize_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_equalsize_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_equalsize_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_gradient_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_gradient_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_gradient_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_gradient_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_gradient_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_randomnodes_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_randomnodes_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_randomnodes_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_randomnodes_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_randomnodes_15.sh
sleep 2

###
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_distancedensity_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_distancedensity_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_distancedensity_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_distancedensity_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_distancedensity_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_equalsize_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_equalsize_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_equalsize_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_equalsize_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_equalsize_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_gradient_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_gradient_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_gradient_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_gradient_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_gradient_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_randomnodes_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_randomnodes_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_randomnodes_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_randomnodes_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_randomnodes_15.sh
sleep 2



######
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_noattention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_noattention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_noattention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_max_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_noattention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_noattention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_noattention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_mean_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_noattention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_noattention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_noattention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_min_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_noattention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_noattention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_noattention_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_noattention_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_pna_noattention_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_sum_noattention_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_sum_noattention_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_noripple_sum_noattention_7.sh
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

###
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_distancedensity_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_distancedensity_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_distancedensity_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_distancedensity_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_distancedensity_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_equalsize_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_equalsize_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_equalsize_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_equalsize_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_equalsize_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_gradient_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_gradient_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_gradient_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_gradient_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_gradient_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_randomnodes_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_randomnodes_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_randomnodes_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_randomnodes_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_min_noattention_randomnodes_15.sh
sleep 2

###
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_distancedensity_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_distancedensity_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_distancedensity_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_distancedensity_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_distancedensity_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_equalsize_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_equalsize_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_equalsize_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_equalsize_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_equalsize_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_gradient_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_gradient_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_gradient_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_gradient_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_gradient_15.sh
sleep 2

sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_randomnodes_3.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_randomnodes_5.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_randomnodes_7.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_randomnodes_10.sh
sleep 2
sbatch --gres=gpu -t 48:00:00 -p gpu_8 cloth_ripple_sum_noattention_randomnodes_15.sh
sleep 2