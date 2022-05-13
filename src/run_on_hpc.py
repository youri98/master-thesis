#!/usr/bin/env python3
#SBATCH --partition=mcs.gpu.q
#SBATCH --output=openme.out
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --error=slurm-%j.err
#SBATCH --time=2:00:00
#SBATCH --gres

import sys, os
sys.path.append(os.getcwd())

from config import get_params
from main import train_model
import wandb
import torch


if __name__ == '__main__':
    config = get_params()

    # run 1
    config["total_rollouts"] = int(50)
    config["n_workers"] = 16

    train_model(config, run_from_hpc=False)
    wandb.finish()