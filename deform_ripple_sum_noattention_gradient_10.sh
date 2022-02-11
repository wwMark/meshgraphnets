#!/bin/bash
#SBATCH --job-name=deform_noripple_sum_noattention
#SBATCH --partition=gpu_8

i=10
srun --exclusive -N1 -p gpu_8 --gres=gpu python run_model.py --model=deform --mode=all --rollout_split=valid --dataset=deforming_plate  --epochs=25 --trajectories=1000 --num_rollouts=100 --core_model=encode_process_decode --message_passing_aggregator=sum --message_passing_steps=${i} --attention=False --ripple_used=True --ripple_generation=gradient --ripple_generation_number=5 --ripple_node_selection=all --ripple_node_selection_random_top_n=5 --ripple_node_connection=fully_ncross_connected --ripple_node_ncross=3 --use_prev_config=True
