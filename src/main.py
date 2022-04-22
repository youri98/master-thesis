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
from utils import rebin, visualize, setup_environment, plot_score, downscale
import datetime
from models import PPOAgent, RND
from utils import plot_learning_curve
import torch as T

random.seed(42)

ROM = "MontezumaRevenge"


def run_game(env: ALEInterface, track_score: bool = True, record: bool = False, downscaling: int = 5, max_frames: int = 1000):
    actions = env.getMinimalActionSet()
    n_actions = len(actions)

    recording = []
    score = []
    N = 200
    batch_size = 50
    n_epochs = 4
    alpha = .0003

    obs_dim = env.getScreenGrayscale().shape
    print(obs_dim)

    agent = PPOAgent(n_actions=n_actions, batch_size=batch_size,
                     alpha=alpha, n_epochs=n_epochs, input_dims=obs_dim)

    rnd_agent = RND()
    n_games = 1

    figure_file = 'plots/cartpole.png'

    score_history = []
    learn_iters = 0
    avg_score = 0 
    n_steps = 0

    for i in range(n_games):
        #observation = env.reset()
        done = False
        score = 0
        observation = env.getScreenGrayscale()
        downscaled_obs = downscale(observation, 5)
        print(downscaled_obs.shape)
        r_i = 0

        while not done and env.getEpisodeFrameNumber() < max_frames:
            observation = T.tensor(env.getScreenGrayscale(), dtype=T.float)
            observation = T.unsqueeze(observation, dim=0)

            action, prob, reward_i = agent.choose_action(observation)
            reward_e = env.act(action)
            done = env.game_over()

            n_steps += 1
            score += reward_e
            agent.remember(observation, action, prob, reward_e, reward_i, done)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f'episode {i} score: {score} avg_score {avg_score} time stepts {n_steps} learning steps {learn_iters}')

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)



def main():
    env = setup_environment(ROM)
    run_game(env)


if __name__ == '__main__':
    main()
