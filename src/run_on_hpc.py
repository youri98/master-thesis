import sys, os
sys.path.append(os.getcwd())

from config import get_params
from main import train_model
import wandb


if __name__ == '__main__':
    config = get_params()
    train_model(config, run_from_hpc=False)
    wandb.finish()