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
models = ['TM', 'WTM', 'RTM', 'WRTM']
targets = ['rc', 'sine','tri','sqr','rect']
backgrounds = ['uniform','modulated']
amp_rescale = 128*0.6*128 #norm=128*0.6, adding*128

#%%
path = 'sims/exp1_1.5cpd/'
filename = 'exp1_original'
cmp_th = prepare.load_sim_data(path, filename)
cmp_th *= amp_rescale # from rms level to Frobenius norm

#%%
threshold.cmp_th(targets, backgrounds, models, cmp_th, path=str(Path(__file__).parent))

#%%
path = 'sims/exp1_1.5cpd/'
filename = 'exp1_uncertain'
cmp_th_un = prepare.load_sim_data(path, filename)
cmp_th_un *= amp_rescale # from rms level to Frobenius norm

#%%
path = 'sims/exp1_1.5cpd/'
filename = 'exp1_eye'
cmp_th_eye = prepare.load_sim_data(path, filename)
cmp_th_eye *= amp_rescale # from rms level to Frobenius norm

#%%
path = 'sims/exp1_1.5cpd/'
filename = 'exp1_uncertain_eye'
cmp_th_uncertain_eye = prepare.load_sim_data(path, filename)
cmp_th_uncertain_eye *= amp_rescale # from rms level to Frobenius norm

#%%
threshold.cmp_th(targets, backgrounds, models, cmp_th_un, path=str(Path(__file__).parent), theme='uncertain_')

#%%
subject = 'AVE'
sessions = ['AD1','AD2','AZ1','AZ2','CO1', 'CO2']
path = "data/exp1_1.5cpd/"
# prepare.clean_exp_data(path, sessions, amp_rescale)
amps, tar_pre, acc = prepare.load_exp_data(path, sessions)

#%%
threshold.learning(targets, backgrounds, sessions, amps, acc, path_save=str(Path(__file__).parent))

#%%
reload(bootstrap)
# amps, tar_pre, acc = prepare.load_exp_data(path, sessions)
# bootstrap.across_level(sessions, amps, acc, tar_pre, n_samp=1000, path_save=str(Path(__file__).parent))

#%%
reload(threshold)

subject = 'AD'
sessions = ['AD1','AD2']
amps, tar_pre, acc = prepare.load_exp_data(path, sessions)
threshold.psychometric(targets, backgrounds, sessions, subject, amps, acc, path=str(Path(__file__).parent))
subject = 'AZ'
sessions = ['AZ1','AZ2']
amps, tar_pre, acc = prepare.load_exp_data(path, sessions)
threshold.psychometric(targets, backgrounds, sessions, subject, amps, acc, path=str(Path(__file__).parent))
subject = 'CO'
sessions = ['CO1','CO2']
amps, tar_pre, acc = prepare.load_exp_data(path, sessions)
threshold.psychometric(targets, backgrounds, sessions, subject, amps, acc, path=str(Path(__file__).parent))
subject = 'AVE'
sessions = ['AD1','AD2','AZ1','AZ2','CO1', 'CO2']
amps, tar_pre, acc = prepare.load_exp_data(path, sessions)
threshold.psychometric(targets, backgrounds, sessions, subject, amps, acc, path=str(Path(__file__).parent))

#%%
reload(threshold)
ave_diff = summary.ave_background_diff(targets, backgrounds, sessions, path=str(Path(__file__).parent))
threshold.exp_th(targets, backgrounds, sessions, path=str(Path(__file__).parent))

#%%
threshold.exp_cmp(targets, backgrounds, sessions, subject, models, cmp_th, path=str(Path(__file__).parent))

#%%
reload(threshold)
scales, rmse = threshold.fit_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th, path=str(Path(__file__).parent))

threshold.diff_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th, path=str(Path(__file__).parent))

threshold.dd_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th, path=str(Path(__file__).parent))

#%%
reload(threshold)
sessions1 = ['AD1', 'AD2']
subject1 = 'AD'
sessions2 = ['AZ1', 'AZ2']
subject2 = 'AZ'
sessions3 = ['CO1', 'CO2']
subject3 = 'CO'
sessions4 = ['AD1', 'AD2', 'AZ1', 'AZ2', 'CO1', 'CO2']
subject4 = 'AVE'
threshold.exp_th4(targets, backgrounds, sessions1, sessions2, sessions3, sessions4, path=str(Path(__file__).parent))

#%%
threshold.exp_cmp(targets, backgrounds, sessions, subject, models, cmp_th_un, path=str(Path(__file__).parent), theme='uncertain_')

#%%
reload(threshold)
models = ['TM(u)', 'WTM(u)', 'RTM(u)', 'WRTM(u)']
scales, rmse = threshold.fit_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_un, path=str(Path(__file__).parent), theme='uncertain_')

threshold.diff_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_un, path=str(Path(__file__).parent), theme='uncertain_')

threshold.dd_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_un, path=str(Path(__file__).parent), theme='uncertain_')

threshold.detail_dd_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_un, path=str(Path(__file__).parent), theme='uncertain_')

#%%
reload(threshold)
models = ['TM', 'ETM', 'RTM', 'ERTM']
scales, rmse = threshold.fit_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_eye, path=str(Path(__file__).parent), theme='eye_')

threshold.diff_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_eye, path=str(Path(__file__).parent), theme='eye_')

threshold.dd_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_eye, path=str(Path(__file__).parent), theme='eye_')

threshold.detail_dd_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_eye, path=str(Path(__file__).parent), theme='eye_')

#%%
reload(threshold)
models = ['TM(u)', 'ETM(u)', 'RTM(u)', 'ERTM(u)']
scales, rmse = threshold.fit_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_uncertain_eye, path=str(Path(__file__).parent), theme='uncertain_eye_')

threshold.diff_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_uncertain_eye, path=str(Path(__file__).parent), theme='uncertain_eye_')

threshold.dd_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_uncertain_eye, path=str(Path(__file__).parent), theme='uncertain_eye_')

threshold.detail_dd_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th_uncertain_eye, path=str(Path(__file__).parent), theme='uncertain_eye_')

#%%
threshold.cmp_cmp(targets, backgrounds, models, cmp_th, cmp_th_un, theme2='(u)', path=str(Path(__file__).parent))

# %%
sessions1 = ['AD1', 'AD2']
subject1 = 'AD'
sessions2 = ['AZ1', 'AZ2']
subject2 = 'AZ'
threshold.exp_exp(targets, sessions1, subject1, sessions2, subject2, path=str(Path(__file__).parent))

# %%
sessions1 = ['AD1', 'AD2']
subject1 = 'AD'
sessions2 = ['AZ1', 'AZ2']
subject2 = 'AZ'
sessions3 = ['CO1', 'CO2']
subject3 = 'CO'
threshold.exp_exp_exp(targets, sessions1, subject1, sessions2, subject2, sessions3, subject3, path=str(Path(__file__).parent))

#%%
reload(threshold)
models = ['ERTM', 'UERTM']
sessions = ['AD1', 'AD2', 'AZ1', 'AZ2', 'CO1', 'CO2']
scales, rmse = threshold.fit_diff_cmp2_exp(targets, backgrounds, sessions, subject, models, cmp_th_eye,  cmp_th_uncertain_eye, path=str(Path(__file__).parent))

#%%
path = 'sims/exp1_1.5cpd/'
filename = 'exp1_uncertain_low'
cmp_th_un_l = prepare.load_sim_data(path, filename)
cmp_th_un_l *= amp_rescale # from rms level to Frobenius norm
filename = 'exp1_uncertain'
cmp_th_un_m = prepare.load_sim_data(path, filename)
cmp_th_un_m *= amp_rescale # from rms level to Frobenius norm
filename = 'exp1_uncertain_high'
cmp_th_un_h = prepare.load_sim_data(path, filename)
cmp_th_un_h *= amp_rescale # from rms level to Frobenius norm

threshold.cmp_cmp_cmp_cmp(targets, backgrounds, models, cmp_th, cmp_th_un_l, cmp_th_un_m, cmp_th_un_h, theme2='(low u)', theme3='(mid u)', theme4='(high u)', path=str(Path(__file__).parent))

#%%
models = ['TM', 'WTM', 'RTM', 'WRTM', 'ERTM', 'UERTM']
rms = (10*np.array([3.204, 5.717, 1.532, 4.465, 1.663, 1.189])+8*np.array([1.835, 4.082, 1.969, 3.735, 1.603, 0.968]))/18
scale = (10*np.array([0.884, 0.459, 0.610, 0.302, 0.475, 0.654])+8*np.array([0.676, 0.354, 0.462, 0.247, 0.421, 0.803]))/18

reload(threshold)
threshold.rank_rms_scale(models, rms, scale, path=str(Path(__file__).parent))

#%%
reload(threshold)
n_params = np.ones((len(models),))
aic = threshold.akaike_information_criterion(cmp_th, targets, models, n_params, sessions, path=str(Path(__file__).parent))
print(aic)

#%%
reload(threshold)
n_params = np.ones((len(models),))
aic = threshold.akaike_information_criterion(cmp_th_eye, targets, models, n_params, sessions, path=str(Path(__file__).parent))
print(aic)

#%%
reload(threshold)
n_params = np.ones((len(models),))
aic = threshold.akaike_information_criterion(cmp_th_uncertain_eye, targets, models, n_params, sessions, path=str(Path(__file__).parent))
print(aic)

#%%
