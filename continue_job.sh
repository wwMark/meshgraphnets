#!/bin/bash

sbatch --gres=gpu -t 48:00:00 -p gpu_8 task_continue.sh "$1"
