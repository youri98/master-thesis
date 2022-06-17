import sys, os
sys.path.append(os.getcwd())

from config import get_params
from main import noisy_tv, train_model
import wandb
import time

if __name__ == '__main__':
    config = get_params()
    start = time.time()
    train_model(config, add_noisy_tv=True)
    wandb.finish()
    stop = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    print(f"program took {stop}")