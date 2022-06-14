from inspect import modulesbyfile
from config import get_params
import numpy as np
from numpy import concatenate
import gym
from tqdm import tqdm
from ape import APE
from rnd import RND
from logger import Logger
from torch.multiprocessing import Process, Pipe
from runner import Worker
import torch
import wandb
import multiprocessing
import os
import pygad
import time
import gym
import numpy as np
import pygad.torchga
import pygad
import torch
import torch.nn as nn
from multiprocessing import Pool
from utils import *
import pygad.gacnn
from torch.distributions.categorical import Categorical
from collections import deque
from GA import GAfunctions, PooledGA
import globals
import sys, os
import wandb
import time

sys.path.append(os.getcwd())

if __name__ == '__main__':
    start = time.time()

    torch_ga = pygad.torchga.TorchGA(model=globals.agent.current_policy.cpu(), num_solutions=globals.config["n_individuals_per_gen"])
    initial_population = torch_ga.population_weights  # Initial population of network weights

    print(np.concatenate(initial_population).shape)

    ga_instance = PooledGA(num_generations=globals.config["num_generations"],
                        num_parents_mating=globals.config["num_parents_mating"],
                        parent_selection_type=globals.config["parent_selection_type"],
                        crossover_type=globals.config["crossover_type"],
                        mutation_type=globals.config["mutation_type"],
                        mutation_percent_genes=globals.config["mutation_percent_genes"],
                        keep_parents=globals.config["keep_parents"],
                        initial_population=initial_population,
                        fitness_func=GAfunctions.fitness_func,
                        on_generation=GAfunctions.callback_generation,
                        on_parents=GAfunctions.on_parents,
                        on_mutation=GAfunctions.on_mutation,
                        on_fitness=GAfunctions.on_fitness,
                        on_crossover=GAfunctions.on_crossover,
                        on_start=GAfunctions.on_start,
                        save_best_solutions=False) # might use this for GAETL
                        # sol_per_pop=10,
                        # num_genes=300)

    print("ga instantiated")
    globals.pool = Pool(processes=globals.config["n_workers"])
    print(globals.pool)
    ga_instance.run()

    wandb.finish()
    stop = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    print(f"program took {stop}")
    