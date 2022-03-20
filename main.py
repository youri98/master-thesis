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
from utils import rebin, visualize, setup_environment


random.seed(42)

ROM = "MontezumaRevenge"


def run_game(env: ALEInterface, record: bool = False, downscaling: int = 5, max_frames: int = 100):
    actions = env.getMinimalActionSet()
    num_actions = len(actions)

    recording = []
    total_reward = 0
    while not env.game_over() and env.getEpisodeFrameNumber() < max_frames:
        # get state
        observation = env.getScreenGrayscale()
        downscaled_obs = rebin(
            observation, (observation.shape[0] // downscaling, observation.shape[1] // downscaling))  # downscale

        # select action via model
        a = actions[randrange(num_actions)]

        # get reward
        reward = env.act(a)
        total_reward += reward

        if record:
            recording.append(downscaled_obs)

    if record:
        visualize(recording)

    print(f'Episode ended with score: {total_reward}')


def main():
    env = setup_environment(ROM)
    run_game(env)


if __name__ == '__main__':
    main()
