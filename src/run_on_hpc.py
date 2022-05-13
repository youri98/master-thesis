import sys, os
sys.path.append(os.getcwd())

from config import get_params
from main import train_model
import wandb
import time

if __name__ == '__main__':
    config = get_params()
    start = time.time()
    train_model(config, run_from_hpc=False)
    wandb.finish()
    stop = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    print(f"program took {stop}")