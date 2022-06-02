#%%
from matplotlib import rcParams
import numpy as np
from numpy import random
import pandas as pd
import matplotlib as mpl
from matplotlib.pyplot import *
from pathlib import Path
import os
from importlib import reload
import time

from numba import cuda
if cuda.is_available():
    from cupy.cuda import Device
    from cupyx.time import repeat

#%%

base_path = Path(__file__).parent.parent.parent
os.chdir(base_path)

from models import searcher
from plots import eyemove
from data import maps

#%%
rcParams['figure.figsize'] = [8,6]
rcParams['figure.dpi'] = 100
rcParams['savefig.format'] = 'pdf'

#%%
target = pd.read_csv("data/targets/template_v_30m.csv", header=None).values
dp = pd.read_csv("data/dp_map/dp_anqi_30ppd.csv", header=None).values
background_l = 480
background = random.normal(0,1,[background_l, background_l])

loc_target = np.array([background_l/4, background_l*3/4]).astype(int)
trow, tcol = target.shape
srow, scol = background.shape
stimulus = background
stimulus[int(loc_target[0]-trow//2):int(loc_target[0]+trow//2), int(loc_target[1]-tcol//2):int(loc_target[1]+tcol//2)] += target

t_mask = maps.circ_mask(srow, scol, (srow-trow)//2)
s_mask = maps.circ_mask(srow, scol, srow//2)

#%%
reload(searcher)

tic = time.perf_counter()
p_fix, loc_fix, p_map, p_map0 = searcher.ELM(target, t_mask, stimulus, s_mask, dp, loc_prior0=0.1, gamma=0.8, max_fix=5)
toc=time.perf_counter()
print(toc-tic)

#%%
reload(searcher)

tic = time.perf_counter()
with Device(0):
    p_fix, loc_fix, p_map, p_map0 = searcher.ELM_gpu(target, t_mask, stimulus, s_mask, dp, loc_target, loc_prior0=0.1, gamma=0.8, max_fix=5)
toc=time.perf_counter()
print(toc-tic)

# %%
reload(eyemove)

eyemove.print_frames(p_fix, loc_fix, p_map, p_map0, loc_target, str(Path(__file__).parent))
theme = 'ELM'
eyemove.video(theme, str(Path(__file__).parent), 0.5, str(Path(__file__).parent))

# %%

# %%
