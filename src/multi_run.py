from calendar import c
from pathos.multiprocessing import ProcessingPool, ThreadingPool
import sys, os
from config import get_params
from main import train_model
import wandb
import time
from pathos.multiprocessing import ProcessingPool, ThreadingPool
import random

sys.path.append(os.getcwd())



# build a non-blocking processing pool map (i.e. async_map)


# define an 'inner' function 
# def g(x):
#    return int(x * random.random())
 
# # parallelize the inner function
# def h(x):
#    return sum(tmap(g, x))
 
# # define the 'outer' function
# def f(x,y):
#    return x*y

# # define two lists of different lengths
# x = range(10)
# y = range(5)
 
# evaluate in nested parallel (done several times, for effect)
# res1 = amap(f, [h(x),h(x),h(x),h(x),h(x)], y)
# res2 = amap(f, [h(x),h(x),h(x),h(x),h(x)], y)
# res3 = amap(f, [h(x),h(x),h(x),h(x),h(x)], y)
# res4 = amap(f, [h(x),h(x),h(x),h(x),h(x)], y)

# print(res1.get())
# print(res2.get())
# print(res3.get())
# print(res4.get())


# print("hhiii")

def run_once(mem_size):
    config = get_params()
    config["mem_size"] = mem_size
    config["verbose"] = True
    config["fix_beta"] = True
    config["sampling_algo"] = "per"

    
    start = time.time()
    train_model(config)
    wandb.finish()
    stop = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    print(f"program took {stop}")

if __name__ == '__main__':
    mems = [1, 2]
    amap = ProcessingPool().amap
    # build a blocking thread pool map
    tmap = ThreadingPool().map
    res4 = amap(run_once, mems)
    print(res4.get())
