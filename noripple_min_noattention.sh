#!/bin/bash
#SBATCH --job-name=noripple_min_noattention
#SBATCH --partition=gpu_8

i=7
do
	srun --exclusive -N1 -p gpu_8 --gres=gpu python run_model.py --model=cloth --mode=train --rollout_split=valid --epochs=25 --trajectories=1000 --num_rollouts=100 --core_model=encode_decode_process --message_passing_aggregator=min --message_passing_steps=${i} --attention=False --ripple_used=False --ripple_generation=equal_size --ripple_generation_number=1 --ripple_node_selection=random --ripple_node_selection_random_top_n=1 --ripple_node_connection=most_influential --ripple_node_ncross=1
done
