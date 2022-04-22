from ale_py import ALEInterface
from ale_py.roms import Breakout, MontezumaRevenge
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import typing
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
import datetime

def rebin(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    # downscales 2D-array with mean
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def downscale(a: np.ndarray, factor: int) -> np.ndarray:
    shape = (a.shape[0]//factor, a.shape[1]//factor)
    return rebin(a, shape)



def visualize(recording: np.ndarray, fps: int = 120):

    fig = plt.figure(1, figsize=(8, 8))

    def animate(i):
        fig.clf()
        plt.imshow(recording[i], cmap='gray', aspect='equal')
        plt.axis('off')
        fig.canvas.draw()

    anim = animation.FuncAnimation(
        fig, animate, interval=1, frames=len(recording))
    writer = animation.FFMpegWriter(fps=fps)

    print(f"saving {len(recording)//fps} seconds of recording...")
    start = time.time()
    anim.save("recordings/recording.mp4", writer=writer, dpi=50)
    print(f"done saving! It took {time.time() - start} seconds")


def setup_environment(rom: str) -> ALEInterface:
    game_dict = {"MontezumaRevenge": MontezumaRevenge,
                 "Breakout": Breakout}
    ale = ALEInterface()
    ale.loadROM(game_dict[rom])
    return ale

def plot_score(score: list):
    now = datetime.datetime.now()

    fig = plt.figure(1, figsize=(8, 3))
    plt.plot(score)
    plt.ylim(ymin=0)

    plt.savefig(f"figures/{now.hour}{now.minute}-reward")


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)