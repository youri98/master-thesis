from config import get_params
from ape import APE
from logger import Logger
from utils import *
import sys, os
import wandb
import time

sys.path.append(os.getcwd())

config = get_params()
agent = APE(**config)
logger = Logger(agent, **config)
logger.log_config_params()


envs = [make_atari(config["env"], config["max_frames_per_episode"]) for _ in range(config["n_individuals_per_gen"])] # change this outside call    
