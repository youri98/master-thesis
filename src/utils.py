from ale_py import ALEInterface
from ale_py.roms import Breakout, MontezumaRevenge
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import typing
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
import datetime
import cv2
import os
import re

def rebin(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    # downscales 2D-array with mean
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def downscale(a: np.ndarray, factor: int) -> np.ndarray:
    shape = (a.shape[0]//factor, a.shape[1]//factor)
    return rebin(a, shape)

def save_recordings(recordings, input_dims, fps=60):
    for i, record in enumerate(recordings):
        save_recording(record, input_dims, fps=fps, filename=f'output{i}')



def save_recording(recording, input_dims, fps=60, filename='output'):
    #print(f"saving {len(recording)//fps} seconds of recording...")

    start = time.time()
    frame_size = (input_dims[1], input_dims[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(f'recordings/{filename}.mp4', fourcc, fps, frame_size)
    
    # width_pad = (500 - dims[1]) // 2 
    # height_pad = (500 - dims[0]) // 2 

    for image in recording:
        #image = np.pad(image, ((height_pad, height_pad), (width_pad,width_pad)))
        image = np.expand_dims(image, axis=2)
        image = np.tile(image, (1,1,3))

        out.write(image)

    out.release()
    #print(f"done saving! It took {time.time() - start} seconds")


def setup_environment(rom: str) -> ALEInterface:
    game_dict = {"MontezumaRevenge": MontezumaRevenge,
                 "Breakout": Breakout}
    ale = ALEInterface()
    ale.loadROM(game_dict[rom])
    return ale

def plot_score(total_e_score, total_i_score):
        
    figure_file = 'plots/score.png'
    fig = plt.figure(1, figsize=(8, 3))
    #plt.ylim(ymin=0)
    plt.yscale('log')

    for i, (e_score, i_score) in enumerate(zip(total_e_score, total_i_score)):
        x = [i+1 for i in range(len(e_score))]
        plt.plot(x, i_score, color=((i+1)/(len(total_e_score) + 1), 0, 0))
        plt.plot(x, e_score, color=(0, 0, (i+1)/(len(total_e_score) + 1)))

    now = datetime.datetime.now()
    plt.savefig(f"plots/{now.hour}{now.minute}-reward")


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def rename_best_model(dir):
    for folder in ["/actor/", "/predictor/"]:
        p = dir + folder
        model_names = [name for name in os.listdir(p)]
        new_best = model_names[-1]
        prev_best =  list(filter(lambda v: re.match('.*_best', v), model_names))[0]
        if new_best == prev_best:
            continue
        rename = prev_best.replace('_best', '')
        os.rename(p+prev_best, p+rename)
        os.rename(p+new_best, p+new_best+"_best")

def delete_files():
    dir = os.getcwd()

    
    for folder in ["/recordings", "/tmp/actor/", "/tmp/target/", "/tmp/predictor/"]:
        
        p = dir + folder
        model_names = [name for name in os.listdir(p)]