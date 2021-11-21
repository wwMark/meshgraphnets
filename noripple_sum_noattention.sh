#!/bin/bash
#SBATCH --job-name=noripple_sum_noattention
#SBATCH --partition=gpu_8

for i in 1 2 3 4 5 6 7 8 9 10 15
do
	srun --exclusive -N1 -p gpu_8 --gres=gpu python run_model.py --message_passing_aggregator=sum --message_passing_steps=${i} --attention=False --ripple_used=False --epochs=25 --trajectories=1000 --num_rollouts=100
done
