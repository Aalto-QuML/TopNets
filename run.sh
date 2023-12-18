#!/bin/bash
#SBATCH --job-name=TopNNs
#SBATCH --time=01-23:14:00
#SBATCH --account=project_2006852
#SBATCH --mem-per-cpu=15G
#SBATCH --output=Train_topnn_high.out
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1


python -u main_qm9_topnn.py --property homo --wandb online --batch_size 96