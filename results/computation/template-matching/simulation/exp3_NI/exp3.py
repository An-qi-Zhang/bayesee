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

#%%
base_path = Path(__file__).parent.parent.parent
os.chdir(base_path)

from data import prepare
from analysis import bootstrap, summary
from plots import threshold

#%%
def decibel(x):
    return 20*np.log10(x)

#%%
models = ['TM', 'whitening', 'weighting', 'both']
targets = ['rc', 'sine','tri','sqr','rect']
backgrounds = ['uniform','modulated']
amp_rescale = 128*0.6*128

#%%
subject = 'AVE'
sessions = ['AD1','AD2','AZ1','AZ2','CO1', 'CO2']
path = "data/exp1_1.5cpd/"
#prepare.clean_exp_data(path, sessions, amp_rescale)
amps, tar_pre, acc = prepare.load_exp_data(path, sessions)

#%%
models = ['WTM', 'WRTM', 'RTM', 'ERTM', 'UERTM']
backgrounds = ['1/f noise','natural images']

#%%
delta_thresholds = np.zeros((len(backgrounds), len(models)))

cmp_th = prepare.load_sim_data('sims/exp1_1.5cpd/', 'exp1_original')
db_cmp_th = decibel(cmp_th)

#WTM
delta_thresholds[0,0] = (db_cmp_th[0,:]+db_cmp_th[4,:]).mean()/2-(db_cmp_th[1,:]+db_cmp_th[5,:]).mean()/2

#RTM
delta_thresholds[0,2] = (db_cmp_th[0,:]+db_cmp_th[4,:]).mean()/2-(db_cmp_th[2,:]+db_cmp_th[6,:]).mean()/2

#WRTM
delta_thresholds[0,1] = (db_cmp_th[0,:]+db_cmp_th[4,:]).mean()/2-(db_cmp_th[3,:]+db_cmp_th[7,:]).mean()/2

#%%
cmp_th_eye = prepare.load_sim_data('sims/exp1_1.5cpd/', 'exp1_eye')
db_cmp_th_eye = decibel(cmp_th_eye)

delta_thresholds[0,3] = (db_cmp_th[0,:]+db_cmp_th[4,:]).mean()/2-(db_cmp_th_eye[3,:]+db_cmp_th_eye[7,:]).mean()/2

#%%
cmp_th_uncertain_eye = prepare.load_sim_data('sims/exp1_1.5cpd/', 'exp1_uncertain_eye')
db_cmp_th_uncertain_eye = decibel(cmp_th_uncertain_eye)

delta_thresholds[0,4] = (db_cmp_th[0,:]+db_cmp_th[4,:]).mean()/2-(db_cmp_th_uncertain_eye[3,:]+db_cmp_th_uncertain_eye[7,:]).mean()/2

#%%
path = 'sims/exp3_NI/'
var_B = ["1_7_4", "1_7_7", "1_10_3", "1_10_7", "7_3_6", "4_6_5", "6_6_7", "5_10_7", "6_10_5", "8_2_6", "10_2_4", "9_6_4", "10_5_7", "9_9_3", "8_10_7"]

cmp_th_NI = np.zeros((12, len(var_B)))
for b in range(len(var_B)):
    filename = 'eye_cm4_ni_b'+ var_B[b]
    cmp_th_NI[:,b] = prepare.load_sim_data(path, filename).mean(axis=1)
    
db_cmp_th_NI = decibel(cmp_th_NI.mean(axis=1))

# WTM
delta_thresholds[1,0] = (db_cmp_th_NI[0]+db_cmp_th_NI[6])/2-(db_cmp_th_NI[1]+db_cmp_th_NI[7])/2

# RTM
delta_thresholds[1,2] = (db_cmp_th_NI[0]+db_cmp_th_NI[6])/2-(db_cmp_th_NI[2]+db_cmp_th_NI[8])/2

# ERTM
delta_thresholds[1,3] = (db_cmp_th_NI[0]+db_cmp_th_NI[6])/2-(db_cmp_th_NI[5]+db_cmp_th_NI[11])/2

# WRTM
delta_thresholds[1,1] = (db_cmp_th_NI[0]+db_cmp_th_NI[6])/2-(db_cmp_th_NI[3]+db_cmp_th_NI[9])/2

#%%
path = 'sims/exp3_NI/'
var_B = ["1_7_4", "1_7_7", "1_10_3", "1_10_7", "4_6_5", "5_10_7", "6_6_7", "6_10_5", "7_3_6", "8_2_6", "8_10_7",  "9_6_4", "9_9_3", "10_2_4", "10_5_7"]
cmp_th_NI_uncertain = np.zeros((4, len(var_B)))
for b in range(len(var_B)):
    filename = 'uncertain_eye_cm4_ni_b'+ var_B[b]
    cmp_th_NI_uncertain[:,b] = prepare.load_sim_data(path, filename).mean(axis=1)

db_cmp_th_NI_uncertain = decibel(cmp_th_NI_uncertain.mean(axis=1))

#UERTM
delta_thresholds[1,4] = (db_cmp_th_NI[0]+db_cmp_th_NI[6])/2-(db_cmp_th_NI_uncertain[1]+db_cmp_th_NI_uncertain[3])/2

#%%
colors = ['gray', 'k']

reload(threshold)
threshold.delta_cmps_th(backgrounds, models, colors, delta_thresholds, path=str(Path(__file__).parent))

#%%
path = 'sims/exp3_NI/'
var_B = ["000", "164", "178", "1101", "11010", "553", "558",  "5101", "51010", "1013", "1018", "10101", "101010"]

#%%
reload(threshold)
for bin in var_B:
    filename = 'cm4_ni_b' + bin
    cmp_th = prepare.load_sim_data(path, filename)
    cmp_th *= amp_rescale # from rms level to Frobenius norm
    threshold.cmp_th(targets, backgrounds, models, cmp_th, path=str(Path(__file__).parent), theme=filename)

#%%
reload(threshold)
for bin in var_B:
    filename = 'eye_cm4_ni_b' + bin
    cmp_th = prepare.load_sim_data(path, filename)
    cmp_th *= amp_rescale # from rms level to Frobenius norm
    # threshold.cmp_th(targets, backgrounds, models, cmp_th, path=str(Path(__file__).parent), theme=filename)
    threshold.detail_dd_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th, path=str(Path(__file__).parent), theme=bin)

#%%
path = 'sims/exp3_NI/'
var_B = ["000", "164", "178", "1101", "11010", "553", "558",  "5101", "51010", "1013", "1018", "10101", "101010"]

#%%
reload(threshold)
for bin in var_B:
    filename = 'cm4_ni_b' + bin
    cmp_th = prepare.load_sim_data(path, filename)
    cmp_th *= amp_rescale # from rms level to Frobenius norm
    threshold.cmp_th(targets, backgrounds, models, cmp_th, path=str(Path(__file__).parent), theme=filename)

#%%
reload(threshold)
for bin in var_B:
    filename = 'eye_cm4_ni_b' + bin
    cmp_th = prepare.load_sim_data(path, filename)
    cmp_th *= amp_rescale # from rms level to Frobenius norm
    # threshold.cmp_th(targets, backgrounds, models, cmp_th, path=str(Path(__file__).parent), theme=filename)
    threshold.detail_dd_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th, path=str(Path(__file__).parent), theme=bin)

#%%