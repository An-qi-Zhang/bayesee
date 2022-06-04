#%%
from pathlib import Path
from scipy.io import loadmat
import numpy as np
from pprint import pprint as pp
from matplotlib.pyplot import *

#%%
class Subject:
    def __init__(self, blocks, plot_name = 'ANONYMOUS'):
        self.blocks = blocks
        self.plot_name = plot_name
        
    def load_mats(self, file_paths, variables, vars_name=None):
        if hasattr(self, 'file_paths'):
            self.file_paths += file_paths
        else:
            self.file_paths = file_paths
        
        for file_path in file_paths:
            
            if vars_name is None:
                vars_name = variables
            
            if hasattr(self, 'vars'):
                self.vars = self.vars | dict.fromkeys(vars_name)
            else:
                self.vars = dict.fromkeys(vars_name)
        
                
            for i, var in enumerate(vars_name):
                if self.vars[var] is None :
                    self.vars[var] = loadmat(file_path)[variables[i]]
                else:
                    self.vars[var] = np.concatenate((self.vars[var], loadmat(file_path)[variables[i]]), axis=self.vars[var].ndim-1) # combine along the last dimension
    
    def calc_percent_correct(self, b_acc_var, p_acc_var='p_acc'):
        self.vars[p_acc_var] = np.einsum('ijkl->jkl', self.vars[b_acc_var]) / self.vars[b_acc_var].shape[0]
    
    def plot_accuracy(self, x_var, b_acc_var, pargs, p_acc_var='p_acc'):
        n_trials, n_sessions, n_conditions, n_levels = self.vars[b_acc_var].shape
        
        if p_acc_var not in self.vars:
            self.calc_percent_correct(b_acc_var, p_acc_var)
                
        fig, axs = subplots(nrows=n_conditions, ncols=n_sessions, figsize=pargs['figsize'], constrained_layout=True)
        amp_min = self.vars[x_var].min() * 0.9
        amp_max = self.vars[x_var].max() * 1.1

        for i in range(n_sessions):
            for j in range(n_conditions):
                if n_sessions == 1 or n_conditions == 1:
                    a_idx = j if n_sessions == 1 else i
                else:
                    a_idx = j,i
                    
                axs[a_idx].scatter(self.vars[x_var][i,j,:], self.vars[p_acc_var][i,j,:], color='k', label=f'Sess:{i+1}, Cond{j+1}')
                axs[a_idx].legend(loc='best', fontsize=pargs['fontsizes'][1])
                axs[a_idx].set_xlim(amp_min, amp_max)
                axs[a_idx].set_ylim(0.4, 1.1)
                axs[a_idx].tick_params(axis='x', labelsize= pargs['fontsizes'][2])
                axs[a_idx].tick_params(axis='y', labelsize= pargs['fontsizes'][2])
                
            fig.text(0.5, -0.05, pargs['x_label'], ha='center', fontsize=pargs['fontsizes'][0])
            fig.text(-0.05, 0.5, pargs['y_label'], va='center', rotation='vertical', fontsize=pargs['fontsizes'][0])
            fig.text(0.05, 0.9, self.plot_name, fontsize=pargs['fontsizes'][0])

        savefig('plot_accuracy_' + self.plot_name + '.svg', dpi=300, bbox_inches='tight')
        close()
        
    def bin_vars(self, bin_var, n_bins):
        bin_bounds = [np.quantile(self.vars[bin_var], b/n_bins) for b in range(n_bins+1)]
        self.vars["bin_id"] = np.digitize(self.vars[bin_var], bin_bounds)

    
    
# %%


