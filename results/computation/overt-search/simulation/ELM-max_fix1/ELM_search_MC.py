#%%
from matplotlib import rcParams
import numpy as np
from numpy import random
import pandas as pd
from matplotlib.pyplot import *
from pathlib import Path
import os
from importlib import reload
from scipy.signal import correlate2d

from numba import cuda
if cuda.is_available():
    from cupy.cuda import Device

#%%
base_path = Path(__file__).parent.parent.parent
os.chdir(base_path)

from models import searcher
from plots import eyemove
from analysis import eyemove_stats
from data import maps
from operation import filter

#%%
target = pd.read_csv("data/targets/template_v_30m.csv", header=None).values
dp = pd.read_csv("data/dp_map/dp_anqi_30ppd.csv", header=None).values

# %%
trow, tcol = target.shape
background_l = 480
stimulus = random.normal(0,1,[background_l, background_l])
t_mask = maps.circ_mask(background_l, background_l, (background_l-trow)//2)
s_mask = maps.circ_mask(background_l, background_l, background_l//2)
n_mc = 5000

max_fix = 10
dp *= 0.7

#%%
subfolder = '/max_fix10_0.7dp/'
vars1name = "prior_x0"
vars1 = np.linspace(0.3, 0.6, 6)
vars2name = "gamma"
vars2 = np.linspace(0.4, 0.7, 4)
n_vars1 = len(vars1)
n_vars2 = len(vars2)
np.save(str(Path(__file__).parent)+subfolder+'var1_'+vars1name, vars1)
np.save(str(Path(__file__).parent)+subfolder+'var2_'+vars2name, vars2)

#%%
loc_target = -1 * np.ones((n_vars1, n_vars2, n_mc, 2))
p_fix = np.zeros((n_vars1, n_vars2, n_mc, max_fix+1))
loc_fix = np.zeros((n_vars1, n_vars2, n_mc, max_fix+1, 2))
p_map0 = np.zeros((n_vars1, n_vars2, n_mc, max_fix+1))

for i_coef_d in np.arange(n_vars1):
    for i_gamma in np.arange(n_vars2):
        var1 = vars1[i_coef_d]
        var2 = vars2[i_gamma]
        for i_mc in np.arange(n_mc):
            if random.random() > 0.5:
                loc_candidates = np.where(t_mask==1)
                rand_id = random.randint(len(loc_candidates[0]))
                l_t = [loc_candidates[0][rand_id], loc_candidates[1][rand_id]]
                loc_target[i_coef_d, i_gamma, i_mc,:] = l_t
            else:
                l_t = None

            with Device(0):
                p_fix[i_coef_d, i_gamma, i_mc,:], loc_fix[i_coef_d, i_gamma, i_mc,:,:], _, p_map0[i_coef_d, i_gamma, i_mc,:] = searcher.ELM_gpu(target, t_mask, stimulus, s_mask, dp, l_t, gamma=var2, max_fix=max_fix, loc_prior0=var1)

np.save(str(Path(__file__).parent)+subfolder+'loc_target', loc_target)
np.save(str(Path(__file__).parent)+subfolder+'p_fix', p_fix)
np.save(str(Path(__file__).parent)+subfolder+'loc_fix', loc_fix)
np.save(str(Path(__file__).parent)+subfolder+'p_map0', p_map0)

#%%
vars1 = np.load(str(Path(__file__).parent)+subfolder+'var1_'+vars1name+'.npy')
vars2 = np.load(str(Path(__file__).parent)+subfolder+'var2_'+vars2name+'.npy')
loc_target = np.load(str(Path(__file__).parent)+subfolder+'loc_target.npy')
p_fix = np.load(str(Path(__file__).parent)+subfolder+'p_fix.npy')
loc_fix = np.load(str(Path(__file__).parent)+subfolder+'loc_fix.npy')
p_map0 = np.load(str(Path(__file__).parent)+subfolder+'p_map0.npy')

n_vars1 = len(vars1)
n_vars2 = len(vars2)

# %%
reload(eyemove)
reload(eyemove_stats)

window = filter.Gaussian(24, 6)

mat_accuracy = np.zeros((n_vars1, n_vars2))
mat_search_length = np.zeros((n_vars1, n_vars2))
mat_fix_cost = np.zeros((n_vars1, n_vars2))
for i in range(n_vars1):
    for j in range(n_vars2):
        mc_loc_target = loc_target[i,j,:,:]
        mc_p_fix = p_fix[i,j,:]
        mc_loc_fix = loc_fix[i,j,:,:,:]
        mc_p_map0 = p_map0[i,j,:]

        n_fix = eyemove_stats.n_decision(mc_p_fix)
        distances = eyemove_stats.distances(mc_p_fix, mc_loc_fix, mc_loc_target)
        accuracy = eyemove_stats.accuracy(mc_p_fix, mc_loc_fix, mc_loc_target, mc_p_map0)
        p_loc_fix = eyemove_stats.p_loc_fix(background_l, background_l, mc_p_fix, mc_loc_fix)
        ps_loc_fix = eyemove_stats.ps_loc_fix(background_l, background_l, mc_p_fix, mc_loc_fix, mc_loc_target, mc_p_map0)
        p_n2_err_dfix = eyemove_stats.p_n2_err_dfix(background_l, background_l, mc_p_fix, mc_loc_fix, mc_loc_target, mc_p_map0)
        search_lengths = eyemove_stats.search_lengths(mc_p_fix, mc_loc_fix, mc_loc_target, mc_p_map0)
        fix_costs = eyemove_stats.fix_costs(mc_p_fix, mc_loc_fix, mc_loc_target, mc_p_map0)

        p_loc_fix[background_l//2, background_l//2]=0
        fil_p_loc_fix = correlate2d(p_loc_fix, window, mode='same')
        fil_p_loc_fix /= fil_p_loc_fix.max()

        eyemove.perf_portfolio("ELM_{}{:.2g}_{}{:.2g}".format(vars1name, vars1[i], vars2name, vars2[j]), max_fix, n_fix, distances, accuracy, fil_p_loc_fix, ps_loc_fix, p_n2_err_dfix, search_lengths, fix_costs, t_mask, str(Path(__file__).parent))

        mat_accuracy[i, j] = (accuracy>0).sum()/len(accuracy)
        mat_search_length[i,j] = search_lengths[0]
        mat_fix_cost[i,j] = fix_costs[0]

eyemove.vars2d_lines("ELM_accuracy", vars1, vars2, mat_accuracy, vars1name, vars2name, path_save=str(Path(__file__).parent))

eyemove.vars2d_lines("ELM_ln_search_length", vars1, vars2, np.log(mat_search_length), vars1name, vars2name, path_save=str(Path(__file__).parent))

eyemove.vars2d_lines("ELM_ln_fix_cost", vars1, vars2, np.log(mat_fix_cost), vars1name, vars2name, path_save=str(Path(__file__).parent))

eyemove.vars2d_lines("ELM_length_per_fix", vars1, vars2, mat_search_length/mat_fix_cost, vars1name, vars2name, path_save=str(Path(__file__).parent))

# %%
eyemove.vars2d_surface("ELM_accuracy_3D", vars1, vars2, mat_accuracy, vars1name, vars2name, ((15, 75),(-25,0)), path_save=str(Path(__file__).parent))

eyemove.vars2d_surface("ELM_ln_search_length_3D", vars1, vars2, np.log(mat_search_length), vars1name, vars2name, path_save=str(Path(__file__).parent))

eyemove.vars2d_surface("ELM_ln_fix_cost_3D", vars1, vars2, np.log(mat_fix_cost), vars1name, vars2name, path_save=str(Path(__file__).parent))

eyemove.vars2d_surface("ELM_length_per_fix_3D", vars1, vars2, mat_search_length/mat_fix_cost, vars1name, vars2name, ((15,75), (15, 105)), path_save=str(Path(__file__).parent))

#%%