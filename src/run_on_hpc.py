import sys, os
sys.path.append(os.getcwd())

from main import main
import wandb
import time

if __name__ == '__main__':
    start = time.time()
    main()
    wandb.finish()
    stop = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    print(f"program took {stop}")