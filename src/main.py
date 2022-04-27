import sys
import numpy as np
from ale_py import ALEInterface
from ale_py.roms import Breakout, MontezumaRevenge
import random
from random import randrange
import matplotlib.pyplot as plt
import cv2
from IPython.display import HTML
import matplotlib.animation as animation
from collections import deque
from queue import Queue
from IPython import display
from utils import rebin, setup_environment, plot_score, downscale, save_recordings, rename_best_model
import datetime
from models import PPOAgent, RND
from utils import plot_learning_curve
import torch as T
from tqdm import tqdm
import time
from statistics import mean
import os 
import re
from pathlib import Path


random.seed(42)

ROM = "MontezumaRevenge"


def run_game(env: ALEInterface, track_score: bool = True, record: bool = False, downscaling: int = 5, max_frames: int = np.inf):
    actions = env.getMinimalActionSet()
    n_actions = len(actions)

    recording = []
    N = 50
    batch_size = 50
    n_epochs = 4
    alpha = .0003

    obs_dim = env.getScreenGrayscale().shape

    agent = PPOAgent(n_actions=n_actions, batch_size=batch_size,
                     alpha=alpha, n_epochs=n_epochs, input_dims=obs_dim)

    n_games = 5


    e_score_history = []
    i_score_history = []
    learn_iters = 0
    avg_score = 0 
    n_steps = 0
    best_score = 0
    dims = env.getScreenGrayscale().shape
    fps = 5
    recordings = []
    recording = []
    total_i_score = []
    total_e_score = []


    open('log.txt', 'w').close()

    for i in tqdm(range(n_games)):
        #observation = env.reset()
        done = False
        game_frames = 0
        env.reset_game()

        while not done and game_frames < max_frames:
            observation = env.getScreenGrayscale()

            if record:
                recording.append(observation)

            observation = T.tensor(observation, dtype=T.float)
            observation = T.unsqueeze(observation, dim=0)


            action, prob, reward_i = agent.choose_action(observation)
            reward_e = env.act(action)
            done = env.game_over()

            e_score_history.append(reward_e)
            i_score_history.append(reward_i)
            agent.remember(observation, action, prob, reward_e, reward_i, done)
            
            n_steps += 1
            game_frames += 1
            if n_steps % N == 0:
                #print(learn_iters)
                agent.learn(verbose=True)
                learn_iters += 1
            # TODO: save checkpoint

        recordings.append(recording)
        total_e_score.append(e_score_history)
        total_i_score.append(i_score_history)

        recording = []

        avg_score = np.mean(e_score_history[-100:])

        if avg_score >= best_score:
            agent.save_models()
            dir = os.getcwd()
            dir +=  '/tmp'

            if i != 0:
                rename_best_model(dir)
                                
            best_score = avg_score

        print(f'episode {i} \ne_score: {mean(e_score_history)} \ni_score: {mean(i_score_history)} \navg_score {avg_score} \ntime steps {n_steps} \nlearning steps {learn_iters}')
        e_score_history, i_score_history = [], []

    if record:
        save_recordings(recordings, dims)

    plot_score(total_e_score, total_i_score)



def main():
    env = setup_environment(ROM)
    run_game(env, record=True)


if __name__ == '__main__':
    main()
