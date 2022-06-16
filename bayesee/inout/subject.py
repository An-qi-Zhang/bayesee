#%%
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import norm
from scipy.optimize import minimize, curve_fit
import numpy as np
from pprint import pprint as pp
from matplotlib.pyplot import *

#%%
from bayesee.operation.dp import *
from bayesee.operation.mathfunc import *

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
                self.vars = dict.fromkeys(vars_name) | self.vars
            else:
                self.vars = dict.fromkeys(vars_name)
                
            for i, var in enumerate(vars_name):
                if self.vars[var] is None:
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
    
    def bootstrap_neg_ll(self, x_var, t_pre_var, b_acc_var, inits, file, n_samp=1000):
        # inits: alpha, beta, gamma
        n_trials, n_sessions, n_conditions, n_levels = self.vars[b_acc_var].shape
        
        alpha = np.zeros((n_conditions,n_samp))
        beta = np.zeros((n_conditions,n_samp))
        gamma = np.zeros((n_conditions,n_samp))
        
        o = np.zeros((n_conditions,))
        p = np.zeros((n_conditions,))
        q = np.zeros((n_conditions,))
        
        for i in range(n_conditions):
            amp = np.repeat(self.vars[x_var][:,i,:].flatten(order='F'), self.vars[t_pre_var].shape[0])
            stimulus = self.vars[t_pre_var][:,:,i,:].flatten(order='F')
            response = (self.vars[t_pre_var]==self.vars[b_acc_var])[:,:,i,:].flatten(order='F')
            o[i], p[i], q[i] = minimize(neg_ll_glm_disc, inits[i], args=(amp, stimulus, response), method='SLSQP', bounds=((1e-4, self.vars[x_var][:,i,:].max()), (0.5,4), (0, 1))).x
                
        for s in range(n_samp):
            for i in range(n_conditions):
                rand_amp = None
                for j in range(n_levels):
                    amp = np.repeat(self.vars[x_var][:,i,j].flatten(order='F'), self.vars[t_pre_var].shape[0])
                    stimulus = self.vars[t_pre_var][:,:,i,j].flatten(order='F')
                    response = (self.vars[t_pre_var]==self.vars[b_acc_var])[:,:,i,j].flatten(order='F')
                    rand_idx = np.random.choice(len(amp), len(amp), replace=True)
                    
                    if rand_amp is None:
                        rand_amp, rand_stimulus, rand_response = amp[rand_idx], stimulus[rand_idx], response[rand_idx]
                    else:
                        rand_amp = np.concatenate([rand_amp, amp[rand_idx]])
                        rand_stimulus = np.concatenate([rand_stimulus,stimulus[rand_idx]])
                        rand_response = np.concatenate([rand_response,response[rand_idx]])
                    
                alpha[i,s], beta[i,s], gamma[i,s] = minimize(neg_ll_glm_disc, inits[i], args=(rand_amp, rand_stimulus, rand_response), method='SLSQP', bounds=((1e-4,self.vars[x_var][:,i,:].max()), (0.5,4), (0, 1))).x
                
            
        np.savez(file, alpha=alpha, beta=beta, a=o, b=p, c=q, gamma=gamma)
        
        return alpha, beta, gamma
    
    def bootstrap_curve(self, x_var, b_acc_var, file, p_acc_var='p_acc', n_samp=1000):
        # inits: alpha, beta, gamma
        n_trials, n_sessions, n_conditions, n_levels = self.vars[b_acc_var].shape
        
        alpha = np.zeros((n_conditions,n_samp))
        beta = np.zeros((n_conditions,n_samp))
        gamma = np.zeros((n_conditions,n_samp))
        
        if p_acc_var not in self.vars:
            self.calc_percent_correct(b_acc_var, p_acc_var)
            
        for s in range(n_samp):
            for i in range(n_conditions):
                rand_amp = None
                norm_amp = self.vars[x_var][:,i,:].max() / 3 # proper range
                for j in range(n_levels):
                    amp = self.vars[x_var][:,i,j].flatten()
                    p_acc = self.vars[p_acc_var][:,i,j].flatten()
                    rand_idx = np.random.choice(len(amp), len(amp), replace=True)
                    
                    if rand_amp is None:
                        rand_amp, rand_p_acc = amp[rand_idx], p_acc[rand_idx]
                    else:
                        rand_amp = np.concatenate([rand_amp, amp[rand_idx]])
                        rand_p_acc = np.concatenate([rand_p_acc,p_acc[rand_idx]])
                
                (alpha[i,s], beta[i,s], gamma[i,s]), _ = curve_fit(p_acc_glm, rand_amp/norm_amp, rand_p_acc, p0=(1,1,0), bounds=((0, 0.5, -1), (self.vars[x_var][:,i,:].max()/ norm_amp, 4, 1)))
                    
                alpha[i,s] *= norm_amp # compensate normalization
            
        np.savez(file, alpha=alpha, beta=beta, gamma=gamma, a=alpha.mean(axis=1), b=beta.mean(axis=1), c=gamma.mean(axis=1))
                
        return alpha, beta, gamma
    
    def plot_psycho_fit(self, x_var, b_acc_var, pargs, file=None, p_acc_var='p_acc'):
        n_trials, n_sessions, n_conditions, n_levels = self.vars[b_acc_var].shape
        
        if p_acc_var not in self.vars:
            self.calc_percent_correct(b_acc_var, p_acc_var)
            
        bs_neg_ll = np.load(file)
        alpha, beta, gamma, o, p, q = bs_neg_ll['alpha'], bs_neg_ll['beta'], bs_neg_ll['gamma'], bs_neg_ll['a'], bs_neg_ll['b'], bs_neg_ll['c']
        
        fig, axs = subplots(nrows=1, ncols=n_conditions, figsize=pargs['figsize'], constrained_layout=True)
        amp_min = gamma.min()
        amp_max = self.vars[x_var].max() * 1.1
        amp_fit = np.linspace(amp_min, amp_max, 500)
        
        for i in range(n_conditions):
            axs[i].scatter(self.vars[x_var][:,i,:], self.vars[p_acc_var][:,i,:], color='k', label=f'Cond{i+1}')
            axs[i].legend(loc='best', fontsize=pargs['fontsizes'][1])
            axs[i].set_xlim(amp_min, amp_max)
            axs[i].set_ylim(0.4, 1.1)
            axs[i].tick_params(axis='x', labelsize= pargs['fontsizes'][2])
            axs[i].tick_params(axis='y', labelsize= pargs['fontsizes'][2])
            
            a = o[i]
            b = p[i]
            c = q[i]
            
            axs[i].scatter(amp_fit, (norm.cdf(0.5*(amp_fit/a)**b-c)+1-norm.cdf(-0.5*(amp_fit/a)**b-c))/2.0, color='b')
            axs[i].axvline(x=a, color='r', alpha=0.5)
                
        fig.text(0.5, -0.05, pargs['x_label'], ha='center', fontsize=pargs['fontsizes'][0])
        fig.text(-0.05, 0.5, pargs['y_label'], va='center', rotation='vertical', fontsize=pargs['fontsizes'][0])
        fig.text(0.25, 0.9, self.plot_name, fontsize=pargs['fontsizes'][0])

        savefig('plot_fit_' + file.name + '.svg', dpi=300, bbox_inches='tight')
        close()
        
        return alpha, beta, gamma
    
    def bin_vars(self, bin_var, n_bins):
        bin_bounds = [np.quantile(self.vars[bin_var], b/n_bins) for b in range(n_bins+1)]
        self.vars["bin_id"] = np.digitize(self.vars[bin_var], bin_bounds)
        return self.vars["bin_id"]

    def bin_bootstrap_neg_ll(self, bin_var, n_bins, x_var, t_pre_var, b_acc_var, inits, file, n_samp=1000):
        # inits: alpha, beta, gamma
        n_trials, n_sessions, n_conditions, n_levels = self.vars[b_acc_var].shape
        
        b_idx = self.bin_vars(bin_var, n_bins)
        
        alpha = np.zeros((n_conditions,n_samp, n_bins))
        beta = np.zeros((n_conditions,n_samp, n_bins))
        gamma = np.zeros((n_conditions,n_samp, n_bins))
        
        o = np.zeros((n_conditions, n_bins))
        p = np.zeros((n_conditions, n_bins))
        q = np.zeros((n_conditions, n_bins))
        
        for b in range(n_bins):
            for i in range(n_conditions):
                amp = np.repeat(self.vars[x_var][np.newaxis,:,:,:], self.vars[t_pre_var].shape[0], axis=0)[:,:,i,:][b_idx[:,:,i,:]==b+1].flatten()
                stimulus = self.vars[t_pre_var][:,:,i,:][b_idx[:,:,i,:]==b+1].flatten()
                response = (self.vars[t_pre_var]==self.vars[b_acc_var])[:,:,i,:][b_idx[:,:,i,:]==b+1].flatten()
                o[i,b], p[i,b], q[i,b] = minimize(neg_ll_glm_disc, inits[b][i], args=(amp, stimulus, response), method='SLSQP', bounds=((1e-4, self.vars[x_var][:,i,:].max()), (0.5,4), (0, 1))).x
        
        for b in range(n_bins):
            for s in range(n_samp):
                for i in range(n_conditions):
                    rand_amp = None
                    for j in range(n_levels):
                        amp = np.repeat(self.vars[x_var][np.newaxis,:,:,:], self.vars[t_pre_var].shape[0], axis=0)[:,:,i,j][b_idx[:,:,i,j]==b+1].flatten()
                        stimulus = self.vars[t_pre_var][:,:,i,j][b_idx[:,:,i,j]==b+1].flatten()
                        response = (self.vars[t_pre_var]==self.vars[b_acc_var])[:,:,i,j][b_idx[:,:,i,j]==b+1].flatten()
                        rand_idx = np.random.choice(len(amp), len(amp), replace=True)
                        
                        if rand_amp is None:
                            rand_amp, rand_stimulus, rand_response = amp[rand_idx], stimulus[rand_idx], response[rand_idx]
                        else:
                            rand_amp = np.concatenate([rand_amp, amp[rand_idx]])
                            rand_stimulus = np.concatenate([rand_stimulus,stimulus[rand_idx]])
                            rand_response = np.concatenate([rand_response,response[rand_idx]])
                    
                    alpha[i,s,b], beta[i,s,b], gamma[i,s,b] = minimize(neg_ll_glm_disc, inits[b][i], args=(rand_amp, rand_stimulus, rand_response), method='SLSQP', bounds=((1e-4, self.vars[x_var][:,i,:].max()), (0.5,4), (0, 1))).x
                    
        np.savez(file, alpha=alpha, beta=beta, gamma=gamma, a=o, b=p, c=q, n_bins=n_bins)
        
        return alpha, beta, gamma
        
    def bin_bootstrap_curve(self, bin_var, n_bins, x_var, b_acc_var, file=None, p_acc_var='p_acc', n_samp=1000):
        n_trials, n_sessions, n_conditions, n_levels = self.vars[b_acc_var].shape
        
        b_idx = self.bin_vars(bin_var, n_bins)
        
        alpha = np.zeros((n_conditions,n_samp, n_bins))
        beta = np.zeros((n_conditions,n_samp, n_bins))
        gamma = np.zeros((n_conditions,n_samp, n_bins))
        
        if p_acc_var not in self.vars:
            self.calc_percent_correct(b_acc_var, p_acc_var)
            
        for b in range(n_bins):
            for s in range(n_samp):
                for i in range(n_conditions):
                    rand_amp = None
                    norm_amp = self.vars[x_var][:,i,:].max() / 3 # proper range
                    for j in range(n_levels):
                        amp = np.repeat(self.vars[x_var][np.newaxis,:,:,:], self.vars[b_acc_var].shape[0], axis=0)[:,:,i,j][b_idx[:,:,i,j]==b+1].flatten()
                        p_acc = np.repeat(self.vars[p_acc_var][np.newaxis,:,:,:], self.vars[b_acc_var].shape[0], axis=0)[:,:,i,j][b_idx[:,:,i,j]==b+1].flatten()
                        rand_idx = np.random.choice(len(amp), len(amp), replace=True)
                        
                        if rand_amp is None:
                            rand_amp, rand_p_acc = amp[rand_idx], p_acc[rand_idx]
                        else:
                            rand_amp = np.concatenate([rand_amp, amp[rand_idx]])
                            rand_p_acc = np.concatenate([rand_p_acc,p_acc[rand_idx]])
                    
                    (alpha[i,s,b], beta[i,s,b], gamma[i,s,b]), _ = curve_fit(p_acc_glm, rand_amp/norm_amp, rand_p_acc, p0=(1,1,0), bounds=((0, 0.5, -1), (self.vars[x_var][:,i,:].max()/ norm_amp, 4, 1)))
                    
                    alpha[i,s,b] *= norm_amp # compensate normalization
            
        np.savez(file, alpha=alpha, beta=beta, gamma=gamma, a=alpha.mean(axis=1), b=beta.mean(axis=1), c=gamma.mean(axis=1), n_bins=n_bins)
        
        return alpha, beta, gamma
    
    def plot_bin_fit(self, bin_var, n_bins, x_var, b_acc_var, pargs, file, p_acc_var='p_acc'):
        n_trials, n_sessions, n_conditions, n_levels = self.vars[b_acc_var].shape
        
        if p_acc_var not in self.vars:
            self.calc_percent_correct(b_acc_var, p_acc_var)
        
        b_idx = self.bin_vars(bin_var, n_bins)
        
        bin_bs_neg_ll = np.load(file)
        alpha, beta, gamma, o, p, q, pre_n_bins = bin_bs_neg_ll['alpha'], bin_bs_neg_ll['beta'], bin_bs_neg_ll['gamma'], bin_bs_neg_ll['a'], bin_bs_neg_ll['b'], bin_bs_neg_ll['c'], bin_bs_neg_ll['n_bins']

        fig, axs = subplots(nrows=n_conditions, ncols=n_bins, figsize=pargs['figsize'], constrained_layout=True)
        amp_min = gamma.min()
        amp_max = self.vars[x_var].max() * 1.1
        amp_fit = np.linspace(amp_min, amp_max, 500)
        
        for j in range(n_bins):
            for i in range(n_conditions):
                selected_p_acc = np.repeat(self.vars['p_acc'][np.newaxis,:,:,:], self.vars['t_pre'].shape[0], axis=0)[:,:,i,:][b_idx[:,:,i,:]==j+1].flatten()
                weight = np.array([sum(selected_p_acc == p_acc) for p_acc in self.vars['p_acc'][:,i,:].flatten()]) / 2
                axs[i,j].scatter(self.vars[x_var][:,i,:].flatten(), self.vars[p_acc_var][:,i,:].flatten(), s=weight, color='k', label=f'Cond{i+1} Bin{j+1}')
                axs[i,j].legend(loc='best', fontsize=pargs['fontsizes'][1])
                axs[i,j].set_xlim(amp_min, amp_max)
                axs[i,j].set_ylim(0.4, 1.1)
                axs[i,j].tick_params(axis='x', labelsize= pargs['fontsizes'][2])
                axs[i,j].tick_params(axis='y', labelsize= pargs['fontsizes'][2])
                
                a = o[i,j].mean()
                b = p[i,j].mean()
                c = q[i,j].mean()
                
                axs[i,j].scatter(amp_fit, (norm.cdf(0.5*(amp_fit/a)**b-c)+1-norm.cdf(-0.5*(amp_fit/a)**b-c))/2.0, color='b')
                axs[i,j].axvline(x=a, color='r', alpha=0.5)
                
        fig.text(0.5, -0.05, pargs['x_label'], ha='center', fontsize=pargs['fontsizes'][0])
        fig.text(-0.05, 0.5, pargs['y_label'], va='center', rotation='vertical', fontsize=pargs['fontsizes'][0])
        fig.text(0.1, 0.9, self.plot_name, fontsize=pargs['fontsizes'][0])

        savefig('plot_fit_' + file.name + '.svg', dpi=300, bbox_inches='tight')
        close()
        
        return alpha, beta, gamma
        
    def plot_bin_threshold(self, bin_var, n_bins, x_var, b_acc_var, pargs, file, p_acc_var='p_acc'):
        n_trials, n_sessions, n_conditions, n_levels = self.vars[b_acc_var].shape
        
        b_idx = self.bin_vars(bin_var, n_bins)
        
        bin_bs_neg_ll = np.load(file)
        alpha, beta, gamma, o, p, q, pre_n_bins = bin_bs_neg_ll['alpha'], bin_bs_neg_ll['beta'], bin_bs_neg_ll['gamma'], bin_bs_neg_ll['a'], bin_bs_neg_ll['b'], bin_bs_neg_ll['c'], bin_bs_neg_ll['n_bins']
                
        db_th = decibel(o)
        
        fig, ax = subplots(figsize=pargs['figsize'], constrained_layout=True)
        
        low_spat_sim = np.array([self.vars[bin_var][:,:,0,:][b_idx[:,:,0,:]==b+1].mean() for b in range(n_bins)])
        high_spat_sim = np.array([self.vars[bin_var][:,:,1,:][b_idx[:,:,1,:]==b+1].mean() for b in range(n_bins)])
        db_CI68_low_spat_sim = alpha.std(axis=1)[0,:]
        db_CI68_high_spat_sim = alpha.std(axis=1)[1,:]
        ax.errorbar(low_spat_sim, db_th[0,:], yerr=db_CI68_low_spat_sim, fmt='kx', mfc='w', capsize=pargs['fontsizes'][1], markersize=pargs['fontsizes'][0], label='low amp_sim')
        ax.errorbar(high_spat_sim, db_th[1,:], yerr=db_CI68_high_spat_sim, fmt='ko', mfc='w', capsize=pargs['fontsizes'][1], markersize=pargs['fontsizes'][0], label='high amp_sim')
        plot(low_spat_sim, db_th[0,:], 'k-')
        plot(high_spat_sim,db_th[1,:], 'k-')
    
        ax.legend(loc='best', fontsize=pargs['fontsizes'][1])
        ax.tick_params(axis='x', labelsize= pargs['fontsizes'][2])
        ax.tick_params(axis='y', labelsize= pargs['fontsizes'][2])
    
        fig.text(0.5, -0.05, pargs['x_label'], ha='center', fontsize=pargs['fontsizes'][0])
        fig.text(-0.05, 0.5, pargs['y_label'], va='center', rotation='vertical', fontsize=pargs['fontsizes'][0])
        fig.text(0.15, 0.9, self.plot_name, fontsize=pargs['fontsizes'][0])

        savefig('plot_threshold_' + file.name + '.svg', dpi=300, bbox_inches='tight')
        close()
        
        return alpha, beta, gamma
            
# %%


