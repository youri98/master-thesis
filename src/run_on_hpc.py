#!/usr/bin/env python3
#SBATCH --partition=mcs.default.q
#SBATCH --output=openme.out

import sys, os
sys.path.append(os.getcwd())
from config import get_params
from main import train_model
import wandb

if __name__ == '__main__':
    config = get_params()

    # run 1
    config["total_rollouts_per_env"] = int(1000)
    config["algo"] = "RND"

    train_model()
    wandb.finish()