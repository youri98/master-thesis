#!/usr/bin/env python3
#SBATCH --partition=mcs.default.q
#SBATCH --output=openme.out
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --error=slurm-%j.err
#SBATCH --time=2:00:00


import sys, os
sys.path.append(os.getcwd())
from common.config import get_params
from main import train_model
import wandb

if __name__ == '__main__':
    config = get_params()

    # run 1
    config["total_rollouts_per_env"] = int(1000)
    config["algo"] = "RND"
    config["n_workers"] = 16

    train_model()
    wandb.finish()