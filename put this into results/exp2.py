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
targets = ['sine','tri','sqr','rect']
backgrounds = ['uniform','modulated']
amp_rescale = 60 * 0.6 * 128

#%%
path = 'sims/exp2_3cpd/'
filename = 'exp2_original'
cmp_th = prepare.load_sim_data(path, filename)
cmp_th *= amp_rescale # from rms level to Frobenius norm

#%%
threshold.cmp_th(targets, backgrounds, models, cmp_th, path=str(Path(__file__).parent))

#%%
path = 'sims/exp2_3cpd/'
filename = 'exp2_uncertain'
cmp_th_un = prepare.load_sim_data(path, filename)
cmp_th_un *= amp_rescale # from rms level to Frobenius norm

#%%
threshold.cmp_th(targets, backgrounds, models, cmp_th_un, path=str(Path(__file__).parent), theme='uncertain_')

#%%
path = 'sims/exp2_3cpd/'
filename = 'exp2_eye'
cmp_th_eye = prepare.load_sim_data(path, filename)
cmp_th_eye *= amp_rescale # from rms level to Frobenius norm

#%%
path = 'sims/exp2_3cpd/'
filename = 'exp2_uncertain_eye'
cmp_th_uncertain_eye = prepare.load_sim_data(path, filename)
cmp_th_uncertain_eye *= amp_rescale # from rms level to Frobenius norm

#%%
subject = 'AVE'
sessions = ['AZ1', 'AZ2', 'CO1', 'CO2']
path = "data/exp2_3cpd/"
#prepare.clean_exp_data(path, sessions, sidelength)
amps, tar_pre, acc = prepare.load_exp_data(path, sessions)

#%%
threshold.learning(targets, backgrounds, sessions, amps, acc, path_save=str(Path(__file__).parent))

#%%
reload(bootstrap)
# subject = 'AZ'
# sessions = ['AZ1','AZ2']
# amps, tar_pre, acc = prepare.load_exp_data(path, sessions)
# bootstrap.across_level(sessions, amps, acc, tar_pre, n_samp=1000, path_save=str(Path(__file__).parent))
# subject = 'CO'
# sessions = ['CO1','CO2']
# amps, tar_pre, acc = prepare.load_exp_data(path, sessions)
# bootstrap.across_level(sessions, amps, acc, tar_pre, n_samp=1000, path_save=str(Path(__file__).parent))
# subject = 'AVE'
# sessions = ['AZ1','AZ2','CO1', 'CO2']
# amps, tar_pre, acc = prepare.load_exp_data(path, sessions)
# bootstrap.across_level(sessions, amps, acc, tar_pre, n_samp=1000, path_save=str(Path(__file__).parent))

#%%
reload(threshold)

subject = 'AZ'
sessions = ['AZ1','AZ2']
amps, tar_pre, acc = prepare.load_exp_data(path, sessions)
threshold.psychometric(targets, backgrounds, sessions, subject, amps, acc, path=str(Path(__file__).parent))
subject = 'CO'
sessions = ['CO1','CO2']
amps, tar_pre, acc = prepare.load_exp_data(path, sessions)
threshold.psychometric(targets, backgrounds, sessions, subject, amps, acc, path=str(Path(__file__).parent))
subject = 'AVE'
sessions = ['AZ1','AZ2','CO1', 'CO2']
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

#%%
sessions1 = ['AD1', 'AD2']
subject1 = 'AD'
sessions2 = ['AZ1', 'AZ2']
subject2 = 'AZ'
threshold.exp_exp(targets, sessions1, subject1, sessions2, subject2, path=str(Path(__file__).parent))

# %%
sessions2 = ['AZ1', 'AZ2']
subject2 = 'AZ'
sessions3 = ['CO1', 'CO2']
subject3 = 'CO'
threshold.exp_exp_exp(targets, sessions1, subject1, sessions2, subject2, sessions3, subject3, path=str(Path(__file__).parent))

#%%
reload(threshold)
sessions1 = ['AZ1', 'AZ2']
subject1 = 'AZ'
sessions2 = ['CO1', 'CO2']
subject2 = 'CO'
sessions3 = ['AZ1', 'AZ2', 'CO1', 'CO2']
subject3 = 'AVE'
threshold.exp_th3(targets, backgrounds, sessions1, sessions2, sessions3, path=str(Path(__file__).parent))

#%%
reload(threshold)
models = ['ERTM', 'UERTM']
sessions = ['AZ1', 'AZ2', 'CO1', 'CO2']
scales, rmse = threshold.fit_diff_cmp2_exp(targets, backgrounds, sessions, subject, models, cmp_th_eye,  cmp_th_uncertain_eye, path=str(Path(__file__).parent))

#%%
path = 'sims/exp2_3cpd/'
filename = 'exp2_uncertain_low'
cmp_th_un_l = prepare.load_sim_data(path, filename)
cmp_th_un_l *= amp_rescale # from rms level to Frobenius norm
filename = 'exp2_uncertain'
cmp_th_un_m = prepare.load_sim_data(path, filename)
cmp_th_un_m *= amp_rescale # from rms level to Frobenius norm
filename = 'exp2_uncertain_high'
cmp_th_un_h = prepare.load_sim_data(path, filename)
cmp_th_un_h *= amp_rescale # from rms level to Frobenius norm

threshold.cmp_cmp_cmp_cmp(targets, backgrounds, models, cmp_th, cmp_th_un_l, cmp_th_un_m, cmp_th_un_h, theme2='(low u)', theme3='(mid u)', theme4='(high u)', path=str(Path(__file__).parent))

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