from matplotlib import rcParams
import numpy as np
from numpy import random
import pandas as pd
import matplotlib as mpl
from matplotlib.pyplot import *
from pathlib import Path
import os
from importlib import reload
from datetime import datetime

from numba import cuda
if cuda.is_available():
    from cupy.cuda import Device
    from cupyx.time import repeat

#%%
base_path = Path(__file__).parent.parent.parent
os.chdir(base_path)

from models import matcher
from analysis import DVC, psychophysics
from imaging import interaction

#%%
rcParams['figure.figsize'] = [8,6]
rcParams['figure.dpi'] = 100
rcParams['savefig.format'] = 'pdf'

#%%
def show_time(x):
    print("\t\t", x.strftime("%H:%M:%S-%m/%d/%Y"))
    
#%%
ppd = 60
n_trials = 1000
size_ratio = 2
u_backgrounds = 128
sd_backgrounds = 128 * 0.2

#%%
rc = pd.read_csv("data/targets/template_rc.csv", header=None).values
sine = pd.read_csv("data/targets/template_sine_1p5cpd.csv", header=None).values
rect = pd.read_csv("data/targets/template_rect_1p5cpd.csv", header=None).values

#%%
backgrounds = np.random.normal(u_backgrounds,sd_backgrounds,(n_trials, rc.shape[0]*size_ratio, rc.shape[1]*size_ratio))
backgrounds2 = np.random.normal(u_backgrounds,sd_backgrounds,(n_trials, rc.shape[0]*size_ratio, rc.shape[1]*size_ratio))
img_sd = np.ones((backgrounds.shape[1], backgrounds.shape[2])) * sd_backgrounds

#%% rc
template = rc
amp = 38.46
um = 0.083 * ppd

responses = np.zeros((2,n_trials))
abs_responses = np.zeros((2,n_trials))
dps = np.zeros(2)

for i in range(n_trials):
    responses[0,i] = matcher.uncertain_dot_TM_gpu(template,backgrounds[i,:,:]-u_backgrounds, amp, img_sd, um)

    img_present = interaction.add_to_point(amp*template, backgrounds2[i,:,:], (backgrounds2.shape[1]-1)/2, (backgrounds2.shape[2]-1)/2)

    responses[1,i] = matcher.uncertain_dot_TM_gpu(template,img_present-u_backgrounds,amp, img_sd, um)

    abs_responses[0,i] = matcher.uncertain_dot_TM_abs_gpu(template,backgrounds[i,:,:]-u_backgrounds, amp, img_sd, um)

    abs_responses[1,i] = matcher.uncertain_dot_TM_abs_gpu(template,img_present-u_backgrounds,amp, img_sd, um)

dps[0] = psychophysics.dprime_res(responses[0,:], responses[1,:])

dps[1] = psychophysics.dprime_res(abs_responses[0,:], abs_responses[1,:])


np.savez('responses_rc', responses=responses, abs_responses=abs_responses, template=template, amp=amp, dps=dps, um=um)

#%% sine
template = sine
amp = 41.90
um = 0.083 * ppd

responses = np.zeros((2,n_trials))
abs_responses = np.zeros((2,n_trials))
dps = np.zeros(2)

for i in range(n_trials):
    responses[0,i] = matcher.uncertain_dot_TM_gpu(template,backgrounds[i,:,:]-u_backgrounds, amp, img_sd, um)

    img_present = interaction.add_to_point(amp*template, backgrounds2[i,:,:], (backgrounds2.shape[1]-1)/2, (backgrounds2.shape[2]-1)/2)

    responses[1,i] = matcher.uncertain_dot_TM_gpu(template,img_present-u_backgrounds,amp, img_sd, um)

    abs_responses[0,i] = matcher.uncertain_dot_TM_abs_gpu(template,backgrounds[i,:,:]-u_backgrounds, amp, img_sd, um)

    abs_responses[1,i] = matcher.uncertain_dot_TM_abs_gpu(template,img_present-u_backgrounds,amp, img_sd, um)

dps[0] = psychophysics.dprime_res(responses[0,:], responses[1,:])

dps[1] = psychophysics.dprime_res(abs_responses[0,:], abs_responses[1,:])


np.savez('responses_sine', responses=responses, abs_responses=abs_responses, template=template, amp=amp, dps=dps, um=um)

#%% rect
template = rect
amp = 56.82
um = 0.083 * ppd

responses = np.zeros((2,n_trials))
abs_responses = np.zeros((2,n_trials))
dps = np.zeros(2)

for i in range(n_trials):
    responses[0,i] = matcher.uncertain_dot_TM_gpu(template,backgrounds[i,:,:]-u_backgrounds, amp, img_sd, um)

    img_present = interaction.add_to_point(amp*template, backgrounds2[i,:,:], (backgrounds2.shape[1]-1)/2, (backgrounds2.shape[2]-1)/2)

    responses[1,i] = matcher.uncertain_dot_TM_gpu(template,img_present-u_backgrounds,amp, img_sd, um)

    abs_responses[0,i] = matcher.uncertain_dot_TM_abs_gpu(template,backgrounds[i,:,:]-u_backgrounds, amp, img_sd, um)

    abs_responses[1,i] = matcher.uncertain_dot_TM_abs_gpu(template,img_present-u_backgrounds,amp, img_sd, um)

dps[0] = psychophysics.dprime_res(responses[0,:], responses[1,:])

dps[1] = psychophysics.dprime_res(abs_responses[0,:], abs_responses[1,:])

np.savez('responses_rect', responses=responses, abs_responses=abs_responses, template=template, amp=amp, dps=dps, um=um)

# %%
