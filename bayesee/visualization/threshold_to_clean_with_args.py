#%%
import numpy as np
from matplotlib import container
from matplotlib.pyplot import *

from numba import njit
from math import sqrt
import scipy.optimize as opt
from copy import copy

#%%
from bayesee.operation import mathfunc

def decibel_error(x, dx):
    return np.stack((-20*np.log10(1-dx/x), 20*np.log10(1+dx/x)), axis=0)

def load_ths(path, subject):
    exp_ths = np.load(path+'/bs_ths{}.npy'.format(subject))

    CI68 = np.zeros((exp_ths.shape[1],))
    for i in range(exp_ths.shape[1]):
        ths = exp_ths[:,i]
        CI68[i] = CI_width(ths[ths!=0])

    db_exp_th = decibel(exp_th)
    db_CI68 = decibel_error(exp_th, CI68)

    return db_exp_th, db_CI68

#%%
def load_bin_ths(path, subject, n_bins):
    exp_ths = np.load(path+'/bs_ths_{}_{}of{}.npy'.format(subject, 1, n_bins))

    db_exp_ths = np.zeros((n_bins,exp_ths.shape[1]))
    db_CI68s = np.zeros((n_bins, 2, exp_ths.shape[1]))
    CI68 = np.zeros((exp_ths.shape[1],))

    for i in range(n_bins):
        exp_ths = np.load(path+'/bs_ths_{}_{}of{}.npy'.format(subject, i+1, n_bins))
        exp_th = np.mean(exp_ths,axis=0)
        db_exp_ths[i,:] = decibel(exp_th)

        for j in range(exp_ths.shape[1]):
            CI68[j] = CI_width(exp_ths[:,j])

        db_CI68s[i,:,:] = decibel_error(exp_th, CI68)

    return db_exp_ths, db_CI68s

#%%
def efficiency(targets, cmp_th, exp_th, nrows, ncols):
    scales = np.zeros((nrows*ncols))
    for i in range(nrows):
        for j in range(ncols):
            scales[i*nrows+j] = opt.minimize(RMSE, 0, args=(exp_th[:len(targets)], exp_th[len(targets):],cmp_th[i*nrows+j,0,:], cmp_th[i*nrows+j,1,:])).x
    
    return scales

#%%
def efficiency_log(targets, db_cmp_th, db_exp_th, nrows, ncols):
    scales = np.zeros((nrows*ncols))
    for i in range(nrows):
        for j in range(ncols):
            scales[i*nrows+j] = opt.minimize(RMSE_log, 0, args=(db_exp_th[:len(targets)], db_exp_th[len(targets):],db_cmp_th[i*nrows+j,0,:], db_cmp_th[i*nrows+j,1,:])).x
    
    return scales


#%%
def bin_phase_similarity(bin_amps, bin_spatSim, bin_ampSim, subject, path_save='.'):

    n_bins = bin_amps.shape[0]
    db_exp_ths, db_CI68s = load_bin_ths(path_save, subject, n_bins)
    ave_low_ampSim = bin_ampSim[:,:,0].mean()
    ave_high_ampSim = bin_ampSim[:,:,1].mean()
    low_spatSim = bin_spatSim[:,:,0].mean(axis=1)
    high_spatSim = bin_spatSim[:,:,1].mean(axis=1)

    fig = figure(figsize=(10,8))
    subplots_adjust(bottom=0.1, left=0.15)
    
    plot(low_spatSim, db_exp_ths[:,0], 'k-')
    plot(high_spatSim, db_exp_ths[:,1], 'k-')
    
    errorbar(low_spatSim, db_exp_ths[:,0], yerr=db_CI68s[:,:,0].T, fmt='kx', mfc='w', capsize=16, markersize=20, label='ampSim:{:.2f}'.format(ave_low_ampSim))
    errorbar(high_spatSim, db_exp_ths[:,1], yerr=db_CI68s[:,:,1].T, fmt='ko', mfc='w', capsize=16, markersize=20, label='ampSim:{:.2f}'.format(ave_high_ampSim))
    legend(loc='upper right')

    gcf().text(0.5, 0.005, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0.01, 0.5, 'Thresholds (dB)', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_phase_similarity_{}.pdf'.format(subject), bbox_inches='tight')
    savefig(path_save+'/bin_phase_similarity_{}.jpeg'.format(subject), bbox_inches='tight')
    close()
    
    return db_exp_ths

#%%
def bin_phase_similarity_2(bin_amps1, bin_spatSim1, bin_ampSim1, subject1, bin_amps2, bin_spatSim2, bin_ampSim2, subject2, path_save='.'):

    n_bins1 = bin_amps1.shape[0]
    db_exp_ths1, db_CI68s1 = load_bin_ths(path_save, subject1, n_bins1)
    ave_low_ampSim1 = bin_ampSim1[:,:,0].mean()
    ave_high_ampSim1 = bin_ampSim1[:,:,1].mean()
    low_spatSim1 = bin_spatSim1[:,:,0].mean(axis=1)
    high_spatSim1 = bin_spatSim1[:,:,1].mean(axis=1)
    
    n_bins2 = bin_amps2.shape[0]
    db_exp_ths2, db_CI68s2 = load_bin_ths(path_save, subject2, n_bins2)
    ave_low_ampSim2 = bin_ampSim2[:,:,0].mean()
    ave_high_ampSim2 = bin_ampSim2[:,:,1].mean()
    low_spatSim2 = bin_spatSim2[:,:,0].mean(axis=1)
    high_spatSim2 = bin_spatSim2[:,:,1].mean(axis=1)
    
    db_exp_ths_ave, db_CI68s_ave = load_bin_ths(path_save, "AVE", n_bins1)
    
    fig = figure(figsize=(10,8))
    subplots_adjust(bottom=0.1, left=0.15)
    
    plot(low_spatSim1, db_exp_ths_ave[:,0], 'b-')
    plot(high_spatSim1, db_exp_ths_ave[:,1], 'r-')
    
    errorbar(low_spatSim1, db_exp_ths1[:,0], yerr=db_CI68s1[:,:,0].T, fmt='bX', mfc='b', capsize=12, markersize=16, label='ampSim:{:.2f}'.format(ave_low_ampSim1))
    errorbar(high_spatSim1, db_exp_ths1[:,1], yerr=db_CI68s1[:,:,1].T, fmt='bo', mfc='b', capsize=12, markersize=16, label='ampSim:{:.2f}'.format(ave_high_ampSim1))
    
    errorbar(low_spatSim2, db_exp_ths2[:,0], yerr=db_CI68s2[:,:,0].T, fmt='rX', mfc='r', capsize=12, markersize=16)
    errorbar(high_spatSim2, db_exp_ths2[:,1], yerr=db_CI68s2[:,:,1].T, fmt='ro', mfc='r', capsize=12, markersize=16)
    
    ylim([44, 64])
    
    handles, labels = gca().get_legend_handles_labels()
    handles = [copy(i[0]) if isinstance(i, container.ErrorbarContainer) else copy(i) for i in handles]
    for i in handles:
        i.set_markeredgecolor('k')
        i.set_markerfacecolor('k')
    by_label = dict(zip(labels, handles))
    gca().legend(by_label.values(), by_label.keys(),  loc='upper right', fontsize=24)

    gcf().text(0.5, 0.005, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0.01, 0.5, 'Thresholds (dB)', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_phase_similarity_{}_{}.pdf'.format(subject1, subject2), bbox_inches='tight')
    savefig(path_save+'/bin_phase_similarity_{}_{}.jpeg'.format(subject1, subject2), bbox_inches='tight')
    close()
    
    return db_exp_ths1, db_exp_ths2

#%%
def bin_dp_slopes(bin_amps, bin_spatSim, bin_ampSim, bin_tar_pre, bin_tempRes, subject, path_save='.'):

    n_bins = bin_amps.shape[0]
    dps_slopes = np.zeros((n_bins, bin_amps.shape[2]))
    stds_abs_pre = np.zeros((2, len(np.unique(bin_amps)), n_bins, bin_amps.shape[2]))

    fig = figure(2, figsize=(12,8))
    subplots_adjust(bottom=0.075, left=0.075,top=0.85)
    markers = ['o', 's', 'd', 'h', 'X']
    colors= ['r', 'y', 'g', 'b', 'm']
    
    def f(x, a):
        return a * x

    for c in range(bin_amps.shape[2]):
        sca(fig.add_subplot(1, 2, c+1))
        for i in range(n_bins):
            unique_amps = np.unique(bin_amps[i,:,c])
            dps = np.zeros((len(unique_amps),))
            
            for j in range(len(unique_amps)):
                indices = bin_amps[i,:,c]==unique_amps[j]
                tar_pre_indices = np.logical_and(indices, bin_tar_pre[i,:,c] == 1)
                tar_abs_indices = np.logical_and(indices, bin_tar_pre[i,:,c] == 0)
                tar_pre_responses = bin_tempRes[i, tar_pre_indices,c]
                tar_abs_responses = bin_tempRes[i, tar_abs_indices,c]
                n_tar_pre = len(tar_pre_responses)
                n_tar_abs = len(tar_abs_responses)
                ave_std = np.sqrt((n_tar_pre*tar_pre_responses.var() + n_tar_abs*tar_abs_responses.var())/(n_tar_pre+n_tar_abs-2))
                dps[j] = abs(tar_pre_responses.mean()-tar_abs_responses.mean())/ave_std

                stds_abs_pre[0, j, i, c] = tar_abs_responses.std()
                stds_abs_pre[1, j, i, c] = tar_pre_responses.std()

            scatter(unique_amps, dps, s=50, marker=markers[i], edgecolor=colors[i],  facecolor="none", linewidths=3, label=f'bin {i+1}')
            
            dps_slopes[i,c], _ = opt.curve_fit(f, unique_amps, dps)
            x = np.linspace(0, unique_amps.max()*1.2, 500)
            scatter(x, dps_slopes[i,c] * x, s=5, marker='.', edgecolor=colors[i], linewidths=0.5)

            legend(loc='best', fontsize=12)

        gcf().text(0.5, 0.005, 'Amplitude', ha='center', fontsize=24)
        gcf().text(0.01, 0.5, 'd prime', va='center', rotation='vertical', fontsize=24)
        gcf().text(0.3, 0.9, 'aveAmpSim = 0.18', ha='center', fontsize=12)
        gcf().text(0.7, 0.9, 'aveAmpSim = 0.38', ha='center', fontsize=12)

    savefig(path_save+'/bin_dp_slopes_{}.pdf'.format(subject), bbox_inches='tight')
    savefig(path_save+'/bin_dp_slopes_{}.jpg'.format(subject), bbox_inches='tight')
    close()
    
    ave_low_ampSim = bin_ampSim[:,:,0].mean()
    ave_high_ampSim = bin_ampSim[:,:,1].mean()
    low_spatSim = bin_spatSim[:,:,0].mean(axis=1)
    high_spatSim = bin_spatSim[:,:,1].mean(axis=1)
    dps_thresholds = decibel(1/dps_slopes)
    
    fig = figure(figsize=(10,8))
    subplots_adjust(bottom=0.1, left=0.15)
    
    plot(low_spatSim, dps_thresholds[:,0], 'k-')
    plot(high_spatSim, dps_thresholds[:,1], 'k-')
    scatter(low_spatSim, dps_thresholds[:,0], s=300, marker='X', edgecolor='k',  facecolor="k", linewidths=3, label='ampSim:{:.2f}'.format(ave_low_ampSim))
    scatter(high_spatSim, dps_thresholds[:,1], s=300, marker='o', edgecolor='k',  facecolor="k", linewidths=3, label='ampSim:{:.2f}'.format(ave_high_ampSim))
    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.005, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0.01, 0.5, 'Thresholds (dB)', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_th_dp_slopes_{}.pdf'.format(subject), bbox_inches='tight')
    savefig(path_save+'/bin_th_dp_slopes_{}.jpeg'.format(subject), bbox_inches='tight')
    close()
    
    fig = figure(2, figsize=(12,8))
    subplots_adjust(bottom=0.075, left=0.075,top=0.88)
    
    for c in range(bin_amps.shape[2]):
        sca(fig.add_subplot(1, 2, c+1))
        for i in range(n_bins):
            focused_stds1 = stds_abs_pre[0,stds_abs_pre[0,:,i,0]>0, i, c]
            scatter(np.zeros_like(focused_stds1), focused_stds1, s=20, marker='.', edgecolor=colors[i],  facecolor=colors[i], linewidths=1, label=f'bin {i+1}')
            
            focused_stds2 = stds_abs_pre[1,stds_abs_pre[0,:,i,0]>0, i, c]
            scatter(np.ones_like(focused_stds2), focused_stds2, s=20, marker='.', edgecolor=colors[i],  facecolor=colors[i], linewidths=1)
            
            for j in range(len(focused_stds1)):
                plot([0,1], [focused_stds1[j], focused_stds2[j]], '-', linewidth=1, alpha=0.2, color=colors[i])
            
            legend(loc='best', fontsize=10)
            
            parts = violinplot(focused_stds1, positions=[-1+i*0.2], widths=0.2, showextrema=False)
            parts['bodies'][0].set_facecolor(colors[i])
            parts['bodies'][0].set_alpha(1)
            parts = violinplot(focused_stds2, positions=[1.2+i*0.2], widths=0.2, showextrema=False)
            parts['bodies'][0].set_facecolor(colors[i])
            parts['bodies'][0].set_alpha(1)
        
        xlim([-1.2, 2.2])
        xticks([0,1], ['Absent','Present'])
    
    gcf().text(0.3, 0.9, 'aveAmpSim = 0.18', ha='center', fontsize=12)
    gcf().text(0.7, 0.9, 'aveAmpSim = 0.38', ha='center', fontsize=12)
    gcf().text(0.5, 0.005, 'Target', ha='center', fontsize=24)
    gcf().text(0.01, 0.5, 'Standard Deviation of Responses', va='center', rotation='vertical', fontsize=24)
    
    savefig(path_save+'/bin_stds_abs_pre_{}.pdf'.format(subject), bbox_inches='tight')
    savefig(path_save+'/bin_stds_abs_pre_{}.jpeg'.format(subject), bbox_inches='tight')
    close()
    
    return dps_slopes

#%%
def bin_dp_uncertain_slopes(bin_amps, bin_spatSim, bin_ampSim, bin_tar_pre, bin_tempRes, subject, path_save='.'):

    n_bins = bin_amps.shape[0]
    stds_abs_pre = np.zeros((2, len(np.unique(bin_amps)), n_bins, bin_amps.shape[2]))
    dps_uncertain_slopes = np.zeros((n_bins, bin_amps.shape[2], 2))
    dps_amps = np.zeros((bin_amps.shape[0]*bin_amps.shape[1],))

    fig = figure(2, figsize=(12,8))
    subplots_adjust(bottom=0.075, left=0.075,top=0.85)
    markers = ['o', 's', 'd', 'h', 'X']
    colors= ['r', 'y', 'g', 'b', 'm']
    
    def f(x, a, b):
        return np.log((np.exp(a*x)+b)/(1+b))
    
    def f5(x, a, b1, b2, b3, b4, b5):
        y1 = f(x[:bin_amps.shape[1]], a, b1)
        y2 = f(x[bin_amps.shape[1]:bin_amps.shape[1]*2], a, b2)
        y3 = f(x[bin_amps.shape[1]*2:bin_amps.shape[1]*3], a, b3)
        y4 = f(x[bin_amps.shape[1]*3:bin_amps.shape[1]*4], a, b4)
        y5 = f(x[bin_amps.shape[1]*4:], a, b5)
        return np.concatenate((y1, y2, y3, y4, y5), axis=None)

    for c in range(bin_amps.shape[2]):
        sca(fig.add_subplot(1, 2, c+1))
        for i in range(n_bins):
            unique_amps = np.unique(bin_amps[i,:,c])
            dps = np.zeros((len(unique_amps),))
            
            for j in range(len(unique_amps)):
                indices = bin_amps[i,:,c]==unique_amps[j]
                tar_pre_indices = np.logical_and(indices, bin_tar_pre[i,:,c] == 1)
                tar_abs_indices = np.logical_and(indices, bin_tar_pre[i,:,c] == 0)
                tar_pre_responses = bin_tempRes[i, tar_pre_indices,c]
                tar_abs_responses = bin_tempRes[i, tar_abs_indices,c]
                n_tar_pre = len(tar_pre_responses)
                n_tar_abs = len(tar_abs_responses)
                ave_std = np.sqrt((n_tar_pre*tar_pre_responses.var() + n_tar_abs*tar_abs_responses.var())/(n_tar_pre+n_tar_abs-2))
                dps[j] = abs(tar_pre_responses.mean()-tar_abs_responses.mean())/ave_std
                
                stds_abs_pre[0, j, i, c] = tar_abs_responses.std()
                stds_abs_pre[1, j, i, c] = tar_pre_responses.std()

            scatter(unique_amps, dps, s=50, marker=markers[i], edgecolor=colors[i],  facecolor="none", linewidths=3, label=f'bin {i+1}')
            xlim(left=0)
            
            # scaled_unique_amps = unique_amps / unique_amps.max()
            # dps_uncertain_slopes[i,c,:], _ = opt.curve_fit(f, scaled_unique_amps[dps>0], dps[dps>0], bounds=(0, np.inf))
            # dps_uncertain_slopes[i,c,0] /= unique_amps.max()
            
            for k in range(bin_amps.shape[1]):
                dps_amps[i*bin_amps.shape[1]+k] = dps[np.where(unique_amps==bin_amps[i,k,c])]

        scaled_bin_amps = bin_amps[:,:,c] / bin_amps[:,:,c].max()
        slopes, _ = opt.curve_fit(f5, scaled_bin_amps.flatten(), dps_amps, bounds=(0, np.inf))
        dps_uncertain_slopes[:,c,0] = slopes[0] / bin_amps[:,:,c].max()
        
        for i in range(n_bins):
            dps_uncertain_slopes[i,c,1] = slopes[i+1]
            unique_amps = np.unique(bin_amps[i,:,c])
            for j in range(len(unique_amps)):
                x = np.linspace(0, unique_amps.max()*1.2, 500)
                y = np.log((np.exp(dps_uncertain_slopes[i,c,0]*x)+dps_uncertain_slopes[i,c,1])/(1+dps_uncertain_slopes[i,c,1]))
                scatter(x, y, s=5, marker='.', edgecolor=colors[i], linewidths=0.5)
                
            legend(loc='best', fontsize=12)

        gcf().text(0.5, 0.005, 'Amplitude', ha='center', fontsize=24)
        gcf().text(0.01, 0.5, 'd prime', va='center', rotation='vertical', fontsize=24)
        gcf().text(0.3, 0.9, 'aveAmpSim = 0.18', ha='center', fontsize=12)
        gcf().text(0.7, 0.9, 'aveAmpSim = 0.38', ha='center', fontsize=12)

    savefig(path_save+'/bin_dp_uncertain_slopes_{}.pdf'.format(subject), bbox_inches='tight')
    savefig(path_save+'/bin_dp_uncertain_slopes_{}.jpg'.format(subject), bbox_inches='tight')
    close()
    
    ave_low_ampSim = bin_ampSim[:,:,0].mean()
    ave_high_ampSim = bin_ampSim[:,:,1].mean()
    low_spatSim = bin_spatSim[:,:,0].mean(axis=1)
    high_spatSim = bin_spatSim[:,:,1].mean(axis=1)
    dps_thresholds = decibel(np.log((np.exp(1)-1)*dps_uncertain_slopes[:,:,1]+np.exp(1))/dps_uncertain_slopes[:,:,0])
    
    fig = figure(figsize=(10,8))
    subplots_adjust(bottom=0.1, left=0.15)
    
    plot(low_spatSim, dps_thresholds[:,0], 'k-')
    plot(high_spatSim, dps_thresholds[:,1], 'k-')
    scatter(low_spatSim, dps_thresholds[:,0], s=300, marker='X', edgecolor='k',  facecolor="k", linewidths=3, label='ampSim:{:.2f}'.format(ave_low_ampSim))
    scatter(high_spatSim, dps_thresholds[:,1], s=300, marker='o', edgecolor='k',  facecolor="k", linewidths=3, label='ampSim:{:.2f}'.format(ave_high_ampSim))
    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.005, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0.01, 0.5, 'Thresholds (dB)', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_th_dp_uncertain_slopes_{}.pdf'.format(subject), bbox_inches='tight')
    savefig(path_save+'/bin_th_dp_uncertain_slopes_{}.jpeg'.format(subject), bbox_inches='tight')
    close()
    
    fig = figure(2, figsize=(12,8))
    subplots_adjust(bottom=0.075, left=0.075,top=0.88)
    
    for c in range(bin_amps.shape[2]):
        sca(fig.add_subplot(1, 2, c+1))
        for i in range(n_bins):
            focused_stds1 = stds_abs_pre[0,stds_abs_pre[0,:,i,0]>0, i, c]
            scatter(np.zeros_like(focused_stds1), focused_stds1, s=20, marker='.', edgecolor=colors[i],  facecolor=colors[i], linewidths=1, label=f'bin {i+1}')
            
            focused_stds2 = stds_abs_pre[1,stds_abs_pre[0,:,i,0]>0, i, c]
            scatter(np.ones_like(focused_stds2), focused_stds2, s=20, marker='.', edgecolor=colors[i],  facecolor=colors[i], linewidths=1)
            
            for j in range(len(focused_stds1)):
                plot([0,1], [focused_stds1[j], focused_stds2[j]], '-', linewidth=1, alpha=0.2, color=colors[i])
            
            legend(loc='best', fontsize=10)
            
            parts = violinplot(focused_stds1, positions=[-1+i*0.2], widths=0.2, showextrema=False)
            parts['bodies'][0].set_facecolor(colors[i])
            parts['bodies'][0].set_alpha(1)
            parts = violinplot(focused_stds2, positions=[1.2+i*0.2], widths=0.2, showextrema=False)
            parts['bodies'][0].set_facecolor(colors[i])
            parts['bodies'][0].set_alpha(1)
        
        xlim([-1.2, 2.2])
        xticks([0,1], ['Absent','Present'])
    
    gcf().text(0.3, 0.9, 'aveAmpSim = 0.18', ha='center', fontsize=12)
    gcf().text(0.7, 0.9, 'aveAmpSim = 0.38', ha='center', fontsize=12)
    gcf().text(0.5, 0.005, 'Target', ha='center', fontsize=24)
    gcf().text(0.01, 0.5, 'Standard Deviation of Responses', va='center', rotation='vertical', fontsize=24)
    
    savefig(path_save+'/bin_uncertain_stds_abs_pre_{}.pdf'.format(subject), bbox_inches='tight')
    savefig(path_save+'/bin_uncertain_stds_abs_pre_{}.jpeg'.format(subject), bbox_inches='tight')
    close()
    
    return dps_uncertain_slopes

#%%
def neg_accuracy(x, data):
    # x: criterion
    # data[:,0]: template response
    # data[:,1]: target presence
    hit = np.logical_and(data[:,0] > x, data[:,1]==1 ).sum()
    fa = np.logical_and(data[:,0] > x, data[:,1]==0 ).sum()
    return fa - hit

def bin_template_response_blocked_test(bin_amps, bin_tar_pre, bin_spatSim, bin_ampSim, bin_tempRes, subject, path_save='.'):

    n_bins = bin_amps.shape[0]

    ave_low_ampSim = bin_ampSim[:,:,0].mean()
    ave_high_ampSim = bin_ampSim[:,:,1].mean()
    low_spatSim = bin_spatSim[:,:,0].mean(axis=1)
    high_spatSim = bin_spatSim[:,:,1].mean(axis=1)

    all_hit_rate_m = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    all_fa_rate_m = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    all_dp_m = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    
    all_hit_rate_o = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    all_fa_rate_o = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    all_dp_o = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    
    data = np.zeros((bin_ampSim.shape[1], 2)) #template response, target presence
            
    for j in range(bin_ampSim.shape[2]):
        for i in range(n_bins):
            data[:,0] = bin_tempRes[i,:,j]
            data[:,1] = bin_tar_pre[i,:,j]
            tempRes_criterion = data[:,0].mean()
            
            hit_rate = np.logical_and(data[:,0] > tempRes_criterion, data[:,1]==1 ).sum()/ (data[:,1]==1).sum()
            fa_rate = np.logical_and(data[:,0] > tempRes_criterion, data[:,1]==0 ).sum()/(data[:,1]==0 ).sum()

            all_dp_m[i,j] = norm.ppf(hit_rate) - norm.ppf(fa_rate)
            all_hit_rate_m[i,j] = hit_rate
            all_fa_rate_m[i,j] = fa_rate
            
    for j in range(bin_ampSim.shape[2]):
        for i in range(n_bins):
            data[:,0] = bin_tempRes[i,:,j]
            data[:,1] = bin_tar_pre[i,:,j]
            tempRes_criterion = opt.minimize_scalar(neg_accuracy, args=(data),  bounds=(data[:,0].min(), data[:,0].max()), method='bounded', options={'maxiter': 1000, 'disp': 1}).x
            
            hit_rate = np.logical_and(data[:,0] > tempRes_criterion, data[:,1]==1 ).sum()/ (data[:,1]==1).sum()
            fa_rate = np.logical_and(data[:,0] > tempRes_criterion, data[:,1]==0 ).sum()/(data[:,1]==0 ).sum()

            all_dp_o[i,j] = norm.ppf(hit_rate) - norm.ppf(fa_rate)
            all_hit_rate_o[i,j] = hit_rate
            all_fa_rate_o[i,j] = fa_rate
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(low_spatSim, all_dp_m[:,0], s=300, marker='d', edgecolor='k',  facecolor="k", linewidths=3, label='low sim')
    ax.scatter(high_spatSim, all_dp_m[:,1], s=300, marker='o', edgecolor='k', facecolor='k', linewidths=3, label='high sim')

    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Value', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/blocked_m_dp_{}.pdf'.format(subject))
    savefig(path_save+'/blocked_m_dp_{}.jpeg'.format(subject))
    close()
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(low_spatSim, all_dp_o[:,0], s=300, marker='d', edgecolor='r',  facecolor="r", linewidths=3, label='low sim')
    ax.scatter(high_spatSim, all_dp_o[:,1], s=300, marker='o', edgecolor='r', facecolor='r', linewidths=3, label='high sim')

    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Value', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/blocked_o_dp_{}.pdf'.format(subject))
    savefig(path_save+'/blocked_o_dp_{}.jpeg'.format(subject))
    close()
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(low_spatSim, all_fa_rate_m[:,0], s=300, marker='d', edgecolor='k',  facecolor="none", linewidths=3, label='mean')
    ax.scatter(low_spatSim, all_fa_rate_o[:,0], s=300, marker='d', edgecolor='r',  facecolor="none", linewidths=3, label='optimized')

    ax.plot(low_spatSim, all_fa_rate_m[:,0], ':k', alpha=0.5)
    ax.plot(low_spatSim, all_fa_rate_o[:,0], '-.k', alpha=0.5)

    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'False alarm rate | low sim', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/blocked_lowSim_fa_{}.pdf'.format(subject))
    savefig(path_save+'/blocked_lowSim_fa_{}.jpeg'.format(subject))
    close()
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(low_spatSim, 1-all_hit_rate_m[:,0], s=300, marker='d', edgecolor='k',  facecolor="none", linewidths=3, label='mean')
    ax.scatter(low_spatSim, 1-all_hit_rate_o[:,0], s=300, marker='d', edgecolor='r',  facecolor="none", linewidths=3, label='optimized')

    ax.plot(low_spatSim, 1-all_hit_rate_m[:,0], ':k', alpha=0.5)
    ax.plot(low_spatSim, 1-all_hit_rate_o[:,0], '-.k', alpha=0.5)

    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Miss rate | low sim', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/blocked_lowSim_miss_{}.pdf'.format(subject))
    savefig(path_save+'/blocked_lowSim_miss_{}.jpeg'.format(subject))
    close()
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(high_spatSim, all_fa_rate_m[:,1], s=300, marker='d', edgecolor='k',  facecolor="none", linewidths=3, label='mean')
    ax.scatter(high_spatSim, all_fa_rate_o[:,1], s=300, marker='d', edgecolor='r',  facecolor="none", linewidths=3, label='optimized')

    ax.plot(high_spatSim, all_fa_rate_m[:,1], ':k', alpha=0.5)
    ax.plot(high_spatSim, all_fa_rate_o[:,1], '-.k', alpha=0.5)

    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'False alarm rate | high sim', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/blocked_highSim_fa_{}.pdf'.format(subject))
    savefig(path_save+'/blocked_highSim_fa_{}.jpeg'.format(subject))
    close()
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(high_spatSim, 1-all_hit_rate_m[:,1], s=300, marker='d', edgecolor='k',  facecolor="none", linewidths=3, label='mean')
    ax.scatter(high_spatSim, 1-all_hit_rate_o[:,1], s=300, marker='d', edgecolor='r',  facecolor="none", linewidths=3, label='optimized')

    ax.plot(high_spatSim, 1-all_hit_rate_m[:,1], ':k', alpha=0.5)
    ax.plot(high_spatSim, 1-all_hit_rate_o[:,1], '-.k', alpha=0.5)

    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Miss rate | high sim', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/blocked_highSim_miss_{}.pdf'.format(subject))
    savefig(path_save+'/blocked_highSim_miss_{}.jpeg'.format(subject))
    close()
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(low_spatSim, 0.5*(1-all_hit_rate_m[:,0]+all_fa_rate_m[:,0]), s=300, marker='d', edgecolor='k', facecolor='k', linewidths=3, label='MEAN ampSim:{:.2f}'.format(ave_low_ampSim))
    ax.scatter(high_spatSim, 0.5*(1-all_hit_rate_m[:,1]+all_fa_rate_m[:,1]), s=300, marker='o', edgecolor='k',  facecolor="k", linewidths=3, label='MEAN ampSim:{:.2f}'.format(ave_high_ampSim))
    legend(loc='best', fontsize=16)
    ax.scatter(low_spatSim, 0.5*(1-all_hit_rate_o[:,0]+all_fa_rate_o[:,0]), s=300, marker='d', edgecolor='r', facecolor='r', linewidths=3, label='OPT ampSim:{:.2f}'.format(ave_low_ampSim))
    ax.scatter(high_spatSim, 0.5*(1-all_hit_rate_o[:,1]+all_fa_rate_o[:,1]), s=300, marker='o', edgecolor='r',  facecolor="r", linewidths=3, label='OPT ampSim:{:.2f}'.format(ave_high_ampSim))
    legend(loc='best', fontsize=16)

    ax.plot(low_spatSim, 0.5*(1-all_hit_rate_m[:,0]+all_fa_rate_m[:,0]), ':k', alpha=0.5)
    ax.plot(high_spatSim, 0.5*(1-all_hit_rate_m[:,1]+all_fa_rate_m[:,1]), '--k', alpha=0.5)
    ax.plot(low_spatSim, 0.5*(1-all_hit_rate_o[:,0]+all_fa_rate_o[:,0]), ':r', alpha=0.5)
    ax.plot(high_spatSim, 0.5*(1-all_hit_rate_o[:,1]+all_fa_rate_o[:,1]), '--r', alpha=0.5)

    gcf().text(0.5, 0.01, 'Error rate', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Value', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/blocked_error_{}.pdf'.format(subject))
    savefig(path_save+'/blocked_error_{}.jpeg'.format(subject))
    close()

def bin_template_response_blocked(bin_amps, bin_tar_pre, bin_spatSim, bin_ampSim, bin_tempRes, subject, path_save='.'):

    n_bins = bin_amps.shape[0]

    ave_low_ampSim = bin_ampSim[:,:,0].mean()
    ave_high_ampSim = bin_ampSim[:,:,1].mean()
    low_spatSim = bin_spatSim[:,:,0].mean(axis=1)
    high_spatSim = bin_spatSim[:,:,1].mean(axis=1)

    all_hit_rate = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    all_fa_rate = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    all_dp = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    
    data = np.zeros((bin_ampSim.shape[1], 2)) #template response, target presence
    
    for j in range(bin_ampSim.shape[2]):
        for i in range(n_bins):
            data[:,0] = bin_tempRes[i,:,j]
            data[:,1] = bin_tar_pre[i,:,j]
            tempRes_criterion = opt.minimize_scalar(neg_accuracy, args=(data),  bounds=(data[:,0].min(), data[:,0].max()), method='bounded', options={'maxiter': 1000, 'disp': 1}).x
            
            hit_rate = np.logical_and(data[:,0] > tempRes_criterion, data[:,1]==1 ).sum()/ (data[:,1]==1).sum()
            fa_rate = np.logical_and(data[:,0] > tempRes_criterion, data[:,1]==0 ).sum()/(data[:,1]==0 ).sum()

            all_dp[i,j] = norm.ppf(hit_rate) - norm.ppf(fa_rate)
            all_hit_rate[i,j] = hit_rate
            all_fa_rate[i,j] = fa_rate
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(low_spatSim, all_dp[:,0], s=300, marker='d', edgecolor='k',  facecolor="k", linewidths=3, label='low sim')
    ax.scatter(high_spatSim, all_dp[:,1], s=300, marker='o', edgecolor='k', facecolor='k', linewidths=3, label='high sim')

    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Value', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_template_response_blocked_cat_dp_{}.pdf'.format(subject))
    savefig(path_save+'/bin_template_response_blocked_cat_dp_{}.jpeg'.format(subject))
    close()
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(low_spatSim, all_fa_rate[:,0], s=300, marker='d', edgecolor='k',  facecolor="none", linewidths=3, label='false alarm | low sim')
    ax.scatter(low_spatSim, 1-all_hit_rate[:,0], s=300, marker='d', edgecolor='k', facecolor='k', linewidths=3, label='miss | low sim')

    ax.plot(low_spatSim, all_fa_rate[:,0], ':k', alpha=0.5)
    ax.plot(low_spatSim, 1-all_hit_rate[:,0], '-.k', alpha=0.5)

    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Value', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_template_response_blocked_cat_lowSim_{}.pdf'.format(subject))
    savefig(path_save+'/bin_template_response_blocked_cat_lowSim_{}.jpeg'.format(subject))
    close()
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(high_spatSim, all_fa_rate[:,1], s=300, marker='o', edgecolor='k',  facecolor="none", linewidths=3, label='false alarm | high sim')
    ax.scatter(high_spatSim, 1-all_hit_rate[:,1], s=300, marker='o', edgecolor='k', facecolor='k', linewidths=3, label='miss | high sim')
    
    ax.plot(high_spatSim, all_fa_rate[:,1], '--k', alpha=0.5)
    ax.plot(high_spatSim, 1-all_hit_rate[:,1], '-k', alpha=0.5)
    
    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Value', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_template_response_blocked_cat_highSim_{}.pdf'.format(subject))
    savefig(path_save+'/bin_template_response_blocked_cat_highSim_{}.jpeg'.format(subject))
    close()
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(low_spatSim, 0.5*(1-all_hit_rate[:,0]+all_fa_rate[:,0]), s=300, marker='d', edgecolor='k', facecolor='k', linewidths=3, label='ampSim:{:.2f}'.format(ave_low_ampSim))
    ax.scatter(high_spatSim, 0.5*(1-all_hit_rate[:,1]+all_fa_rate[:,1]), s=300, marker='o', edgecolor='k',  facecolor="k", linewidths=3, label='ampSim:{:.2f}'.format(ave_high_ampSim))
    legend(loc='best', fontsize=16)

    ax.plot(low_spatSim, 0.5*(1-all_hit_rate[:,0]+all_fa_rate[:,0]), ':k', alpha=0.5)
    ax.plot(high_spatSim, 0.5*(1-all_hit_rate[:,1]+all_fa_rate[:,1]), '--k', alpha=0.5)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Value', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_template_response_blocked_error_{}.pdf'.format(subject))
    savefig(path_save+'/bin_template_response_blocked_error_{}.jpeg'.format(subject))
    close()

#%%
def bin_template_response_unblocked(bin_amps, bin_tar_pre, bin_spatSim, bin_ampSim, bin_tempRes, subject, path_save='.'):

    n_bins = bin_amps.shape[0]

    ave_low_ampSim = bin_ampSim[:,:,0].mean()
    ave_high_ampSim = bin_ampSim[:,:,1].mean()
    low_spatSim = bin_spatSim[:,:,0].mean(axis=1)
    high_spatSim = bin_spatSim[:,:,1].mean(axis=1)

    all_hit_rate = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    all_fa_rate = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    all_dp = np.zeros((bin_ampSim.shape[0], bin_ampSim.shape[2]))
    
    data = np.zeros((bin_ampSim.shape[0]* bin_ampSim.shape[1], 2)) #template response, target presence

    for j in range(bin_ampSim.shape[2]):
        data[:,0] = bin_tempRes[:,:,j].flatten()
        data[:,1] = bin_tar_pre[:,:,j].flatten()
        tempRes_criterion = opt.minimize_scalar(neg_accuracy, args=(data),  bounds=(data[:,0].min(), data[:,0].max()), method='bounded', options={'maxiter': 1000, 'disp': 1}).x
        for i in range(n_bins):
            hit_rate = np.logical_and(bin_tempRes[i,:,j] > tempRes_criterion, bin_tar_pre[i,:,j]==1 ).sum()/ (bin_tar_pre[i,:,j]==1 ).sum()
            fa_rate = np.logical_and(bin_tempRes[i,:,j] > tempRes_criterion, bin_tar_pre[i,:,j]==0 ).sum()/(bin_tar_pre[i,:,j]==0 ).sum()
            
            all_dp[i,j] = norm.ppf(hit_rate) - norm.ppf(fa_rate)
            all_hit_rate[i,j] = hit_rate
            all_fa_rate[i,j] = fa_rate
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(low_spatSim, all_dp[:,0], s=300, marker='d', edgecolor='k',  facecolor="k", linewidths=3, label='low sim')
    ax.scatter(high_spatSim, all_dp[:,1], s=300, marker='o', edgecolor='k', facecolor='k', linewidths=3, label='high sim')

    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Value', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_template_response_unblocked_cat_dp_{}.pdf'.format(subject))
    savefig(path_save+'/bin_template_response_unblocked_cat_dp_{}.jpeg'.format(subject))
    close()
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(low_spatSim, all_fa_rate[:,0], s=300, marker='d', edgecolor='k',  facecolor="none", linewidths=3, label='false alarm | low sim')
    ax.scatter(low_spatSim, 1-all_hit_rate[:,0], s=300, marker='d', edgecolor='k', facecolor='k', linewidths=3, label='miss | low sim')

    ax.plot(low_spatSim, all_fa_rate[:,0], ':k', alpha=0.5)
    ax.plot(low_spatSim, 1-all_hit_rate[:,0], '-.k', alpha=0.5)

    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Value', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_template_response_unblocked_cat_lowSim_{}.pdf'.format(subject))
    savefig(path_save+'/bin_template_response_unblocked_cat_lowSim_{}.jpeg'.format(subject))
    close()
    
    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(high_spatSim, all_fa_rate[:,1], s=300, marker='o', edgecolor='k',  facecolor="none", linewidths=3, label='false alarm | high sim')
    ax.scatter(high_spatSim, 1-all_hit_rate[:,1], s=300, marker='o', edgecolor='k', facecolor='k', linewidths=3, label='miss | high sim')
    
    ax.plot(high_spatSim, all_fa_rate[:,1], '--k', alpha=0.5)
    ax.plot(high_spatSim, 1-all_hit_rate[:,1], '-k', alpha=0.5)
    
    legend(loc='best', fontsize=16)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Value', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_template_response_unblocked_cat_highSim_{}.pdf'.format(subject))
    savefig(path_save+'/bin_template_response_unblocked_cat_highSim_{}.jpeg'.format(subject))
    close()

    fig = figure(figsize=(10,8))
    ax = gca()
    ax.scatter(low_spatSim, 0.5*(1-all_hit_rate[:,0]+all_fa_rate[:,0]), s=300, marker='d', edgecolor='k', facecolor='k', linewidths=3, label='ampSim:{:.2f}'.format(ave_low_ampSim))
    ax.scatter(high_spatSim, 0.5*(1-all_hit_rate[:,1]+all_fa_rate[:,1]), s=300, marker='o', edgecolor='k',  facecolor="k", linewidths=3, label='ampSim:{:.2f}'.format(ave_high_ampSim))
    legend(loc='best', fontsize=16)

    ax.plot(low_spatSim, 0.5*(1-all_hit_rate[:,0]+all_fa_rate[:,0]), ':k', alpha=0.5)
    ax.plot(high_spatSim, 0.5*(1-all_hit_rate[:,1]+all_fa_rate[:,1]), '--k', alpha=0.5)

    gcf().text(0.5, 0.01, 'Spatial Similarity', ha='center', fontsize=36)
    gcf().text(0, 0.5, 'Value', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/bin_template_response_unblocked_error_{}.pdf'.format(subject))
    savefig(path_save+'/bin_template_response_unblocked_error_{}.jpeg'.format(subject))
    close()


#%%
def learning(targets, backgrounds, sessions, amps, acc, path_save='.'):

    amp_min = amps.min() * 0.8
    amp_max = amps.max() * 1.2
    
    p_acc = acc.mean(axis=0)
    nrows=np.size(p_acc, 0)
    ncols=np.size(p_acc, 1)
    
    fig = figure(figsize=(15,8))
    gs = fig.add_gridspec(nrows,ncols)

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])

            for k in range(len(sessions)):
                if i==nrows-1 and j==ncols-1:
                    ax.scatter(amps[i,j,np.arange(ncols)+5*k], p_acc[i,j,np.arange(ncols)+5*k], label='{}'.format(k+1))
                    legend(loc='lower right')
                else:
                    ax.scatter(amps[i,j,np.arange(ncols)+5*k], p_acc[i,j,np.arange(ncols)+5*k])
                
                xticks(np.linspace(amp_min, amp_max, 5), fontsize=16, rotation=30)
                yticks(np.linspace(0.3, 1, 8))

            if i == 0:
                ax.set_xlabel(targets[j])
                ax.xaxis.set_label_position('top') 
            if i < nrows-1:
                ax.axes.xaxis.set_ticks([])
            
            if j == ncols-1:
                ax.set_ylabel(backgrounds[min(i, nrows-1-i)])
                ax.yaxis.set_label_position('right') 
            
            if j > 0:
                ax.axes.yaxis.set_ticks([])

    gcf().text(0.5, 0.03, 'Target Amplitude', ha='center', fontsize=36)
    gcf().text(0.06, 0.5, 'Percentage Correct', va='center', rotation='vertical', fontsize=36)

    savefig(path_save+'/learning'+'{}.pdf'.format(sessions))
    savefig(path_save+'/learning'+'{}.jpeg'.format(sessions))
    close()

#%%        
def exp_th(targets, backgrounds, sessions, path='.'):

    db_exp_th, db_CI68 = load_ths(path, sessions)

    fig, ax = subplots(figsize=(15,8))
    errorbar(targets, db_exp_th[:len(targets)], yerr=db_CI68[:,:len(targets)],
        fmt='ko', label=backgrounds[0], mew=3, mfc='w', capsize=30, markersize=20)
    errorbar(targets, db_exp_th[len(targets):], yerr=db_CI68[:,len(targets):],
        fmt='ko', label=backgrounds[1], mew=3, capsize=30, markersize=20)
    legend(loc='lower left', fontsize=36)
    ylabel('Amplitude threshold(dB)', fontsize=36)

    # exp1
    xlim([-0.5, 4.5])
    ylim([32, 64]) 
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticks([0.5,1.5,2.5,3.5], minor=True)
    ax.set_yticks([34,38,42,46,50,54,58,62])
    ax.set_yticks([32,36,40,44,48,52,56,60,64], minor=True)

    # exp2
    # xlim([-0.5, 3.5])
    # ylim([28, 48])
    # ax.set_xticks([0,1,2,3])
    # ax.set_xticks([0.5,1.5,2.5], minor=True)
    # ax.set_yticks([30,34,38,42,46])
    # ax.set_yticks([28,32,36,40,44,48], minor=True)

    ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad = 5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
    ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
    ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad =3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
    ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

    savefig(path+'/exp_th'+'{}.pdf'.format(sessions))
    savefig(path+'/exp_th'+'{}.jpeg'.format(sessions))
    close()

#%%
def cmp_th(targets, backgrounds, models, cmp_th, path='.', theme=''):

    nmods = len(models)
    nrows = nmods//2
    ncols = nmods//2

    db_cmp_th = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th[:,0,:] = decibel(cmp_th[nmods:,:])
    db_cmp_th[:,1,:] = decibel(cmp_th[:nmods,:])

    fig = figure(figsize=(32, 16))
    gs = fig.add_gridspec(nrows, ncols)
    colors = ['steelblue', 'firebrick', 'forestgreen', 'indigo']

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])
            
            ax.scatter(targets, db_cmp_th[i*nrows+j,0,:], s=500, marker='D', edgecolor=colors[i*nrows+j],  facecolor="none", linewidths=3,  zorder=1)
            ax.scatter(targets, db_cmp_th[i*nrows+j,1,:], s=500, marker='D', color=colors[i*nrows+j], label=models[i*nrows+j], zorder=1)

            ax.set_ylim(db_cmp_th.flatten().min()-5, db_cmp_th.flatten().max()+5)
            ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
            ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
            ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
            ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

            if j == 1:
                ax.tick_params(axis='y', labelleft=False)
            if i == 0:
                ax.tick_params(axis='x', labelbottom=False)

            legend(loc='lower left', fontsize=36)

    gcf().text(0.002, 0.5, 'Adjusted amplitude threshold(dB)', va='center', rotation='vertical',fontsize=48)
    tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)

    savefig(path+'/{}cmp_th.pdf'.format(theme))
    savefig(path+'/{}cmp_th.png'.format(theme))
    close()

#%%
def psychometric(targets, backgrounds, sessions, subject, amps, acc, path='.'):

    exp_slopes = np.load(path+'/bs_slopes{}.npy'.format(sessions))
    exp_slope = np.mean(exp_slopes,axis=0)

    exp_shapes = np.load(path+'/bs_shapes{}.npy'.format(sessions))
    exp_shape = np.mean(exp_shapes,axis=0)

    exp_criteria = np.load(path+'/bs_criteria{}.npy'.format(sessions))
    exp_criterion = np.mean(exp_criteria,axis=0)

    exp_th = exp_slope

    amp_min = 0
    amp_max = amps.max() * 1.1
    
    p_acc = acc.mean(axis=0)

    nrows=len(backgrounds)
    ncols=len(targets)

    fig = figure(figsize=(15, 8))
    gs = fig.add_gridspec(nrows, ncols)
    
    amp_fit = np.linspace(amp_min, amp_max, 500)

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])
            scatter(amps[i,j,:], p_acc[i,j,:], color='k', marker='x', alpha=0.3)
            scatter(amps[3-i,j,:], p_acc[3-i,j,:], color='k', marker='x', alpha=0.3)

            plot(amp_fit, (norm.cdf(0.5*(amp_fit/exp_slope[ncols*i+j])**exp_shape[ncols*i+j]-exp_criterion[ncols*i+j])+1-norm.cdf(-0.5*(amp_fit/exp_slope[ncols*i+j])**exp_shape[ncols*i+j]-exp_criterion[ncols*i+j]))/2.0, color='b')
            axvline(x=exp_th[ncols*i+j], color='r', alpha=0.5)

            if i==0 and j<3:
                text(amp_min+10, 1.05,r'$\alpha$={:.3f}'.format(exp_slope[ncols*i+j]),  fontsize=16)
                text(amp_min+10, 0.95,r'$\beta$={:.3f}'.format(exp_shape[ncols*i+j]), fontsize=16)
                text(amp_min+10, 0.85,r'$\gamma$={:.3f}'.format(exp_criterion[ncols*i+j]), fontsize=16)
            else:
                text(amp_max*0.35, 0.55,r'$\alpha$={:.3f}'.format(exp_slope[ncols*i+j]), fontsize=16)
                text(amp_max*0.35, 0.45,r'$\beta$={:.3f}'.format(exp_shape[ncols*i+j]), fontsize=16)
                text(amp_max*0.35, 0.35,r'$\gamma$={:.3f}'.format(exp_criterion[ncols*i+j]), fontsize=16)

            xlim([amp_min, amp_max])
            ylim([0.3, 1.1])
            xticks(np.linspace(amp_min, amp_max, 5), fontsize=16, rotation=30)
            yticks(np.linspace(0.3, 1, 8))

            if i == 0:
                ax.set_xlabel(targets[j], fontsize=24)
                ax.xaxis.set_label_position('top') 
            if i < nrows-1:
                ax.axes.xaxis.set_ticks([])
            
            if j == ncols-1:
                ax.set_ylabel(backgrounds[min(i, nrows*2-1-i)], fontsize=24)
                ax.yaxis.set_label_position('right') 
            
            if j > 0:
                ax.axes.yaxis.set_ticks([])

    gcf().text(0.5, 0.01, 'Target Amplitude', ha='center', fontsize=30)
    gcf().text(0.03, 0.5, 'Percentage Correct', va='center', rotation='vertical', fontsize=30)
    gcf().text(0.05, 0.9, subject, ha='center', fontsize=36)

    savefig(path+'/ExpPsychometric'+'{}.pdf'.format(subject))
    savefig(path+'/ExpPsychometric'+'{}.jpeg'.format(subject))
    close()

#%%
def exp_cmp(targets, backgrounds, sessions, subject, models, cmp_th, path='.', theme=''):

    db_exp_th, db_CI68 = load_ths(path, sessions)
        
    nmods = len(models)
    nrows = nmods//2
    ncols = nmods//2

    db_cmp_th = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th[:,0,:] = decibel(cmp_th[nmods:,:])
    db_cmp_th[:,1,:] = decibel(cmp_th[:nmods,:])
            
    fig = figure(figsize=(8,10))
    gs = fig.add_gridspec(nrows, ncols)

    uni_s = 500
    mod_s = 500
    verts = np.zeros((4,4,2))
    verts[0,:,:] = [[0,0],[1,1],[-1,1],[0,0]]
    verts[1,:,:] = [[0,0],[1,1],[1,-1],[0,0]]
    verts[2,:,:] = [[0,0],[-1,1],[-1,-1],[0,0]]
    verts[3,:,:] = [[0,0],[1,-1],[-1,-1],[0,0]]
    colors = ['steelblue', 'gold', 'chartreuse', 'crimson']

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])
            errorbar(targets, db_exp_th[:len(targets)], yerr=db_CI68[:,:len(targets)], fmt='ko', mfc='w', capsize=16, markersize=20)
            errorbar(targets, db_exp_th[len(targets):], yerr=db_CI68[:,len(targets):], fmt='ko', label=subject, capsize=16, markersize=20)
            scatter(targets, db_cmp_th[i*nrows+j,0,:], s=uni_s, marker=verts[i*nrows+j], edgecolor=colors[i*nrows+j], facecolor="none")
            scatter(targets, db_cmp_th[i*nrows+j,1,:], s=mod_s, marker=verts[i*nrows+j], color=colors[i*nrows+j], label=models[i*nrows+j])

            legend(loc='lower left', fontsize=36)
            ylim([db_cmp_th[:,1,-1].min()-10, db_exp_th.max()+10])

            if i < nrows-1:
                ax.axes.xaxis.set_ticks([])
            
            if j > 0:
                ax.axes.yaxis.set_ticks([])
        
    gcf().text(0.5, 0.02, 'target', ha='center',fontsize=36)
    gcf().text(0.02, 0.5, 'Amplitude threshold(dB)', va='center', rotation='vertical',fontsize=36)

    savefig(path+'/{}exp_cmp{}.pdf'.format(theme, sessions))
    savefig(path+'/psychometric'+'{}.jpeg'.format(sessions))
    close()

#%%
def fit_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th, path='.', theme=''):
    db_exp_th, db_CI68 = load_ths(path, sessions)

    nmods = len(models)
    nrows = nmods//2
    ncols = nmods//2

    db_cmp_th = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th[:,0,:] = decibel(cmp_th[nmods:,:])
    db_cmp_th[:,1,:] = decibel(cmp_th[:nmods,:])

    scales = efficiency_log(targets, db_cmp_th, db_exp_th, nrows, ncols)
    rmse = np.zeros_like(scales)

    fig = figure(figsize=(32, 16))
    gs = fig.add_gridspec(nrows, ncols)
    colors = ['steelblue', 'firebrick', 'forestgreen', 'indigo']

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])
            errorbar(targets, db_exp_th[:len(targets)], yerr=db_CI68[:,:len(targets)], fmt='ko', mew=3,  mfc='w', capsize=30, markersize=20, zorder=0)
            if i==0 and j==0:
                errorbar(targets, db_exp_th[len(targets):], yerr=db_CI68[:,len(targets):], fmt='ko', label=subject, mew=3, capsize=30, markersize=20, zorder=0)
            else:
                errorbar(targets, db_exp_th[len(targets):], yerr=db_CI68[:,len(targets):], fmt='ko', mew=3, capsize=30, markersize=20, zorder=0)
            scatter(targets, db_cmp_th[i*nrows+j,0,:]-scales[i*nrows+j], s=500, marker='D', edgecolor=colors[i*nrows+j],  facecolor="none", linewidths=3,  zorder=1)
            scatter(targets, db_cmp_th[i*nrows+j,1,:]-scales[i*nrows+j], s=500, marker='D', color=colors[i*nrows+j], label=models[i*nrows+j], zorder=1)
            
            rmse[i*nrows+j] = RMSE_log(scales[i*nrows+j], db_exp_th[:len(targets)], db_exp_th[len(targets):], db_cmp_th[i*nrows+j,0,:], (db_cmp_th[i*nrows+j,1,:]))

            legend(loc='lower left', fontsize=36)

            # exp1
            # xlim([-0.5, 4.5])
            # ylim([34, 66]) 
            # ax.set_xticks([0,1,2,3,4])
            # ax.set_xticks([0.5,1.5,2.5,3.5], minor=True)
            # ax.set_yticks([34,38,42,46,50,54,58,62,66])
            # ax.set_yticks([36,40,44,48,52,56,60,64], minor=True)

            # exp2
            xlim([-0.5, 3.5])
            ylim([28, 48])
            ax.set_xticks([0,1,2,3])
            ax.set_xticks([0.5,1.5,2.5], minor=True)
            ax.set_yticks([28,32,36,40,44,48])
            ax.set_yticks([30,34,38,42,46], minor=True)

            ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
            ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
            ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
            ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

            if j == 1:
                ax.tick_params(axis='y', labelleft=False)
            if i == 0:
                ax.tick_params(axis='x', labelbottom=False)

    #gcf().text(0.5, 0.02, 'target', ha='center',fontsize=36)
    gcf().text(0.002, 0.5, 'Adjusted amplitude threshold(dB)', va='center', rotation='vertical',fontsize=48)
    gcf().text(0.46, 0.001, 'Targets', va='center', rotation='horizontal',fontsize=48)

    tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)

    savefig(path+'/{}fit_cmp_exp{}.pdf'.format(theme, sessions))
    savefig(path+'/{}fit_cmp_exp{}.png'.format(theme, sessions))
    close()

    return scales, rmse

#%%
def diff_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th, path='.', theme=''):
    db_exp_th, db_CI68 = load_ths(path, sessions)
        
    nmods = len(models)
    nrows = nmods//2
    ncols = nmods//2

    db_cmp_th = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th[:,0,:] = decibel(cmp_th[nmods:,:])
    db_cmp_th[:,1,:] = decibel(cmp_th[:nmods,:])

    fig = figure(figsize=(32, 16))
    gs = fig.add_gridspec(nrows, ncols)

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])
            scatter(targets, db_exp_th[:len(targets)]-db_cmp_th[i*nrows+j,0,:], s=500, marker='^', label=backgrounds[0], edgecolor='k',  facecolor="none", linewidths=3)
            scatter(targets, db_exp_th[len(targets):]-db_cmp_th[i*nrows+j,1,:], s=500, marker='^', color='k', label=backgrounds[1])

            if i==0 and j==0:
                legend(loc='upper right', fontsize=36)

            xlim([-0.5, len(targets)-0.5])
            ylim([-6, 18])
            ax.set_xticks(np.arange(len(targets)))
            ax.set_xticks(np.arange(len(targets)-1)+0.5, minor=True)
            ax.set_yticks([-4,0,4,8,12,16])
            ax.set_yticks([-6,-2,2,6,10,14,18], minor=True)
            axhline(0, linestyle='dotted', linewidth=2, color='k', alpha=0.5)

            ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
            ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
            ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
            ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

            if j == 1:
                ax.tick_params(axis='y', labelleft=False)
            if i == 0:
                ax.tick_params(axis='x', labelbottom=False)

    #gcf().text(0.5, 0.02, 'target', ha='center',fontsize=36)
    gcf().text(0.003, 0.5, 'Amplitude threshold difference (dB)', va='center', rotation='vertical',fontsize=48)
    gcf().text(0.46, 0.001, 'Targets', va='center', rotation='horizontal',fontsize=48)
    tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)

    savefig(path+'/{}diff_cmp_exp{}.pdf'.format(theme, sessions))
    savefig(path+'/{}diff_cmp_exp{}.png'.format(theme, sessions))
    close()

#%%
def dd_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th, path='.', theme=''):
    db_exp_th, db_CI68 = load_ths(path, sessions)
        
    nmods = len(models)
    nrows = nmods//2+1
    ncols = nmods//2+1

    db_cmp_th = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th[:,0,:] = decibel(cmp_th[nmods:,:])
    db_cmp_th[:,1,:] = decibel(cmp_th[:nmods,:])

    fig = figure(figsize=(40,20))
    gs = fig.add_gridspec(nrows, ncols)

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])
            if i < nrows-1 and j < ncols-1:
                ax.scatter(targets, db_exp_th[:len(targets)]-db_cmp_th[i*nmods//2+j,0,:], s=500, marker='^', label=backgrounds[0], edgecolor='k',  facecolor="none", linewidths=3)
                ax.scatter(targets, db_exp_th[len(targets):]-db_cmp_th[i*nmods//2+j,1,:], s=500, marker='^', label=backgrounds[1], color='k')
                ax.text(0.03, 0.85, 'AVE - ' + models[i*nmods//2+j], transform=ax.transAxes, fontsize=48)

                if i==0 and j==0:
                    ax.legend(loc='upper right', fontsize=36)

            else:
                if i == nmods//2:
                    x1 = 1
                    x2 = 0
                    if j != nmods//2:
                        y1 = j
                        y2 = j
                if j == nmods//2:
                    y1 = 1
                    y2 = 0
                    if i != nmods//2:
                        x1 = i
                        x2 = i

                ax.scatter(targets, db_cmp_th[x2*nmods//2+y2,0,:]- db_cmp_th[x1*nmods//2+y1,0,:], s=500, marker='^', edgecolor='k',  facecolor="none", linewidths=3)
                ax.scatter(targets, db_cmp_th[x2*nmods//2+y2,1,:]- db_cmp_th[x1*nmods//2+y1,1,:], s=500, marker='^', color='k')

                # ax.scatter(targets, np.mean(db_cmp_th[x2*nmods//2+y2,:,:]- db_cmp_th[x1*nmods//2+y1,:,:], axis=0), s=500, marker='.', edgecolor='k', facecolor='k')

                ax.text(0.03, 0.85,models[x2*nmods//2+y2]+' - '+models[x1*nmods//2+y1], transform=ax.transAxes, fontsize=48)
                
            # exp1
            xlim([-0.5, 4.5])
            ylim([-10, 18])
            ax.set_xticks([0,1,2,3,4])
            ax.set_xticks([0.5,1.5,2.5,3.5], minor=True)
            ax.set_yticks([-8,-4,0,4,8,12,16])
            ax.set_yticks([-10,-6,-2,2,6,10,14,18], minor=True)

            # exp2
            # xlim([-0.5, 3.5])
            # ax.set_xticks([0,1,2,3])
            # ax.set_xticks([0.5,1.5,2.5], minor=True)
            # ymin = -8
            # ymax= 12
            # ylim([ymin, ymax])
            # ax.set_yticks(np.arange((ymax-ymin)//4+1)*4+ymin)
            # ax.set_yticks(np.arange((ymax-ymin)//4)*4+ymin+2, minor=True)

            # xlim([-0.5, 3.5])
            # ylim([-10, 10])
            # ax.set_xticks([0,1,2,3])
            # ax.set_xticks([0.5,1.5,2.5], minor=True)
            # ax.set_yticks([-8,-4,0,4, 8])
            # ax.set_yticks([-10,-6,-2,2,6, 10], minor=True)
            

            ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
            ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
            ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
            ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

            if j != 0:
                ax.tick_params(axis='y', labelleft=False)
            if i == 0:
                ax.tick_params(axis='x', labelbottom=False)

    #gcf().text(0.5, 0.02, 'target', ha='center',fontsize=36)
    gcf().text(0.003, 0.5, 'amplitude threshold difference (dB)', va='center', rotation='vertical',fontsize=48)
    tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)

    savefig(path+'/{}dd_cmp_exp{}.pdf'.format(theme, sessions))
    savefig(path+'/{}dd_cmp_exp{}.png'.format(theme, sessions))
    close()

#%%
def detail_dd_cmp_exp(targets, backgrounds, sessions, subject, models, cmp_th, path='.', theme=''):
    db_exp_th, db_CI68 = load_ths(path, sessions)
        
    nmods = len(models)

    db_cmp_th = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th[:,0,:] = decibel(cmp_th[nmods:,:])
    db_cmp_th[:,1,:] = decibel(cmp_th[:nmods,:])

    fig = figure(figsize=(12,8))
    ax = gca()

    x1 = 0
    y1 = 1
    x2 = 0
    y2 = 0
    ax.scatter(targets, db_cmp_th[x2*nmods//2+y2,0,:]- db_cmp_th[x1*nmods//2+y1,0,:], s=500, marker='^', edgecolor='blue', label = 'Uni:TM-ETM', facecolor="none", linewidths=3)
    ax.scatter(targets, db_cmp_th[x2*nmods//2+y2,1,:]- db_cmp_th[x1*nmods//2+y1,1,:], s=500, marker='^', color='blue', label='Mod:TM-ETM')

    x1 = 1
    y1 = 1
    x2 = 1
    y2 = 0
    ax.scatter(targets, db_cmp_th[x2*nmods//2+y2,0,:]- db_cmp_th[x1*nmods//2+y1,0,:], s=500, marker='o', edgecolor='red', label = 'Uni:RTM-ERTM', facecolor="none", linewidths=3)
    ax.scatter(targets, db_cmp_th[x2*nmods//2+y2,1,:]- db_cmp_th[x1*nmods//2+y1,1,:], s=500, marker='o', color='red', label='Mod:RTM-ERTM')

    legend(loc='best', fontsize=16)

    # eye
    xlim([-0.5, 4.5])
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticks([0.5,1.5,2.5,3.5], minor=True)

    # xlim([-0.5, 3.5])
    # ax.set_xticks([0,1,2,3])
    # ax.set_xticks([0.5,1.5,2.5], minor=True)

    ymax = 8
    ylim([0,ymax])
    ax.set_yticks(np.arange(ymax+1))
    ax.set_yticks(np.arange(ymax)+0.5, minor=True)
    
    ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
    ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
    ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
    ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

    #gcf().text(0.5, 0.02, 'target', ha='center',fontsize=36)
    ax.set_ylabel('amplitude threshold difference (dB)',fontsize=24)
    tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)

    savefig(path+'/{}detail_dd_cmp_exp{}.pdf'.format(theme, sessions))
    savefig(path+'/{}detail_dd_cmp_exp{}.png'.format(theme, sessions))
    # savefig(path+'/{}detail_dd.png'.format(theme))
    close()


#%%
def cmp_cmp(targets, backgrounds, models, cmp_th1, cmp_th2, theme1='', theme2='', path='.'):
    nmods = len(models)
    nrows = nmods//2
    ncols = nmods//2

    db_cmp_th1 = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th1[:,0,:] = decibel(cmp_th1[nmods:,:])
    db_cmp_th1[:,1,:] = decibel(cmp_th1[:nmods,:])

    db_cmp_th2 = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th2[:,0,:] = decibel(cmp_th2[nmods:,:])
    db_cmp_th2[:,1,:] = decibel(cmp_th2[:nmods,:])
            
    fig = figure(figsize=(8,10))
    gs = fig.add_gridspec(nrows, ncols)

    uni_s = 350
    mod_s = 350
    verts = np.zeros((4,4,2))
    verts[0,:,:] = [[0,0],[1,1],[-1,1],[0,0]]
    verts[1,:,:] = [[0,0],[1,1],[1,-1],[0,0]]
    verts[2,:,:] = [[0,0],[-1,1],[-1,-1],[0,0]]
    verts[3,:,:] = [[0,0],[1,-1],[-1,-1],[0,0]]
    colors1 = ['steelblue', 'gold', 'chartreuse', 'crimson']
    colors2 = ['darkblue', 'peru', 'darkgreen','darkred']

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])
            scatter(targets, db_cmp_th1[i*nrows+j,0,:], s=uni_s, marker=verts[i*nrows+j], color=colors1[i*nrows+j], alpha=0.5)
            scatter(targets, db_cmp_th1[i*nrows+j,1,:], s=mod_s, marker=verts[i*nrows+j], color=colors1[i*nrows+j], label=models[i*nrows+j]+theme1)
            scatter(targets, db_cmp_th2[i*nrows+j,0,:], s=uni_s, marker=verts[i*nrows+j], color=colors2[i*nrows+j], alpha=0.5)
            scatter(targets, db_cmp_th2[i*nrows+j,1,:], s=mod_s, marker=verts[i*nrows+j], color=colors2[i*nrows+j], label=models[i*nrows+j]+theme2)

            legend(loc='lower left', fontsize=36)
            ylim([min(db_cmp_th1.min(),db_cmp_th2.min())-5, max(db_cmp_th1.max(),db_cmp_th2.max())+5])

            if i < nrows-1:
                ax.axes.xaxis.set_ticks([])
            
            if j > 0:
                ax.axes.yaxis.set_ticks([])
        
    gcf().text(0.5, 0.02, 'target', ha='center',fontsize=36)
    gcf().text(0.02, 0.5, 'Amplitude threshold(dB)', va='center', rotation='vertical',fontsize=36)

    savefig(path+'/cmp_cmp_{}vs{}.pdf'.format(theme1, theme2))
    savefig(path+'/cmp_cmp_{}vs{}.png'.format(theme1, theme2))
    close()

#%%
def exp_exp(targets, sessions1, subject1, sessions2, subject2, path='.'):
    db_exp_th1, db_CI681 = load_ths(path, sessions1)

    db_exp_th2, db_CI682 = load_ths(path, sessions2)
            
    fig = figure(figsize=(8,6))

    errorbar(targets, db_exp_th1[:len(targets)], yerr=db_CI681[:,:len(targets)],
                fmt='g.', alpha=0.5, capsize=15, markersize=24)
    errorbar(targets, db_exp_th1[len(targets):], yerr=db_CI681[:,len(targets):],
                fmt='g.', label=subject1, capsize=15, markersize=24)
    
    errorbar(targets, db_exp_th2[:len(targets)], yerr=db_CI682[:,:len(targets)],
                fmt='rs', alpha=0.5, capsize=15, markersize=12)
    errorbar(targets, db_exp_th2[len(targets):], yerr=db_CI682[:,len(targets):],
                fmt='rs', label=subject2, capsize=15, markersize=12)

    legend(loc='lower left', fontsize=36)
    ylim([min(db_exp_th1.min(),db_exp_th2.min())-5, max(db_exp_th1.max(),db_exp_th2.max())+5])
        
    gcf().text(0.5, 0.02, 'target', ha='center',fontsize=36)
    gcf().text(0.02, 0.5, 'Amplitude threshold(dB)', va='center', rotation='vertical',fontsize=36)

    savefig(path+'/exp_exp_{}vs{}.pdf'.format(subject1, subject2))
    savefig(path+'/exp_exp_{}vs{}.png'.format(subject1, subject2))
    close()

#%%
def exp_exp_exp(targets, sessions1, subject1, sessions2, subject2, sessions3, subject3, path='.'):
    db_exp_th1, db_CI681 = load_ths(path, sessions1)

    db_exp_th2, db_CI682 = load_ths(path, sessions2)

    db_exp_th3, db_CI683 = load_ths(path, sessions3)

    fig = figure(figsize=(8,6))

    errorbar(targets, db_exp_th1[:len(targets)], yerr=db_CI681[:,:len(targets)],
                fmt='g.', alpha=0.5, capsize=8, markersize=24)
    errorbar(targets, db_exp_th1[len(targets):], yerr=db_CI681[:,len(targets):],
                fmt='g.', label=subject1, capsize=8, markersize=24)
    
    errorbar(targets, db_exp_th2[:len(targets)], yerr=db_CI682[:,:len(targets)],
                fmt='rs', alpha=0.5, capsize=8, markersize=12)
    errorbar(targets, db_exp_th2[len(targets):], yerr=db_CI682[:,len(targets):],
                fmt='rs', label=subject2, capsize=8, markersize=12)

    errorbar(targets, db_exp_th3[:len(targets)], yerr=db_CI683[:,:len(targets)],
                fmt='bX', alpha=0.5, capsize=8, markersize=12)
    errorbar(targets, db_exp_th3[len(targets):], yerr=db_CI683[:,len(targets):],
                fmt='bX', label=subject3, capsize=8, markersize=12)

    legend(loc='lower left', fontsize=36)
    ylim([min(db_exp_th1.min(),db_exp_th2.min(),db_exp_th3.min())-5, max(db_exp_th1.max(),db_exp_th2.max(),db_exp_th3.max())+5])
        
    gcf().text(0.5, 0.02, 'target', ha='center',fontsize=36)
    gcf().text(0.02, 0.5, 'Amplitude threshold(dB)', va='center', rotation='vertical',fontsize=36)

    savefig(path+'/exp_exp_exp{}vs{}vs{}.pdf'.format(subject1, subject2, subject3))
    savefig(path+'/exp_exp_exp{}vs{}vs{}.png'.format(subject1, subject2, subject3))
    close()

#%%
def cmp_cmp_cmp_cmp(targets, backgrounds, models, cmp_th1, cmp_th2, cmp_th3, cmp_th4, theme1='', theme2='', theme3='', theme4='', path='.'):
    nmods = len(models)
    nrows = nmods//2
    ncols = nmods//2

    db_cmp_th1 = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th1[:,0,:] = decibel(cmp_th1[nmods:,:])
    db_cmp_th1[:,1,:] = decibel(cmp_th1[:nmods,:])

    db_cmp_th2 = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th2[:,0,:] = decibel(cmp_th2[nmods:,:])
    db_cmp_th2[:,1,:] = decibel(cmp_th2[:nmods,:])

    db_cmp_th3 = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th3[:,0,:] = decibel(cmp_th3[nmods:,:])
    db_cmp_th3[:,1,:] = decibel(cmp_th3[:nmods,:])

    db_cmp_th4 = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th4[:,0,:] = decibel(cmp_th4[nmods:,:])
    db_cmp_th4[:,1,:] = decibel(cmp_th4[:nmods,:])

    fig = figure(figsize=(8,10))
    gs = fig.add_gridspec(nrows, ncols)

    uni_s = 350
    mod_s = 350
    verts = np.zeros((4,4,2))
    verts[0,:,:] = [[0,0],[1,1],[-1,1],[0,0]]
    verts[1,:,:] = [[0,0],[1,1],[1,-1],[0,0]]
    verts[2,:,:] = [[0,0],[-1,1],[-1,-1],[0,0]]
    verts[3,:,:] = [[0,0],[1,-1],[-1,-1],[0,0]]
    colors = ['steelblue', 'gold', 'chartreuse', 'crimson']

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])
            scatter(targets, db_cmp_th1[i*nrows+j,0,:], s=uni_s, marker=verts[i*nrows+j], color=colors[0], alpha=0.5)
            scatter(targets, db_cmp_th1[i*nrows+j,1,:], s=mod_s, marker=verts[i*nrows+j], color=colors[0], label=models[i*nrows+j]+theme1)
            scatter(targets, db_cmp_th2[i*nrows+j,0,:], s=uni_s, marker=verts[i*nrows+j], color=colors[1], alpha=0.5)
            scatter(targets, db_cmp_th2[i*nrows+j,1,:], s=mod_s, marker=verts[i*nrows+j], color=colors[1], label=models[i*nrows+j]+theme2)
            scatter(targets, db_cmp_th3[i*nrows+j,0,:], s=uni_s, marker=verts[i*nrows+j], color=colors[2], alpha=0.5)
            scatter(targets, db_cmp_th3[i*nrows+j,1,:], s=mod_s, marker=verts[i*nrows+j], color=colors[2], label=models[i*nrows+j]+theme3)
            scatter(targets, db_cmp_th4[i*nrows+j,0,:], s=uni_s, marker=verts[i*nrows+j], color=colors[3], alpha=0.5)
            scatter(targets, db_cmp_th4[i*nrows+j,1,:], s=mod_s, marker=verts[i*nrows+j], color=colors[3], label=models[i*nrows+j]+theme4)

            legend(loc='lower left', fontsize=36)
            ylim([min(db_cmp_th1.min(),db_cmp_th2.min(), db_cmp_th3.min(), db_cmp_th4.min())-5, max(db_cmp_th1.max(),db_cmp_th2.max(), db_cmp_th3.max(), db_cmp_th4.max())+5])

            if i < nrows-1:
                ax.axes.xaxis.set_ticks([])
            
            if j > 0:
                ax.axes.yaxis.set_ticks([])
        
    gcf().text(0.5, 0.02, 'target', ha='center',fontsize=36)
    gcf().text(0.02, 0.5, 'Amplitude threshold(dB)', va='center', rotation='vertical',fontsize=36)

    savefig(path+'/cmp_cmp_cmp_cmp_{}vs{}vs{}vs{}.pdf'.format(theme1, theme2, theme3, theme4))
    savefig(path+'/cmp_cmp_cmp_cmp_{}vs{}vs{}vs{}.png'.format(theme1, theme2, theme3, theme4))
    close()

#%%
def exp_th4(targets, backgrounds, sessions1, sessions2, sessions3, sessions4, path='.', theme=''):
    db_exp_th1, db_CI681 = load_ths(path, sessions1)

    db_exp_th2, db_CI682 = load_ths(path, sessions2)

    db_exp_th3, db_CI683 = load_ths(path, sessions3)

    db_exp_th4, db_CI684 = load_ths(path, sessions4)
    
    nrows = 2
    ncols = 2

    fig = figure(figsize=(32, 16))
    gs = fig.add_gridspec(nrows, ncols)

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])

            if i==0 and j==0:
                errorbar(targets, db_exp_th1[:len(targets)], yerr=db_CI681[:,:len(targets)], fmt='ko', label=backgrounds[0], mew=3,  mfc='w', capsize=30, markersize=20)
                errorbar(targets, db_exp_th1[len(targets):], yerr=db_CI681[:,len(targets):], fmt='ko', label=backgrounds[1], mew=3, capsize=30, markersize=20)

                legend(loc='lower left', fontsize=36)
            
            elif i==0 and j==1:
                errorbar(targets, db_exp_th2[:len(targets)], yerr=db_CI682[:,:len(targets)], fmt='ko', label=backgrounds[0], mew=3,  mfc='w', capsize=30, markersize=20)
                errorbar(targets, db_exp_th2[len(targets):], yerr=db_CI682[:,len(targets):], fmt='ko', label=backgrounds[1], mew=3, capsize=30, markersize=20)

            elif i==1 and j==0:
                errorbar(targets, db_exp_th3[:len(targets)], yerr=db_CI683[:,:len(targets)], fmt='ko', label=backgrounds[0], mew=3,  mfc='w', capsize=30, markersize=20)
                errorbar(targets, db_exp_th3[len(targets):], yerr=db_CI683[:,len(targets):], fmt='ko', label=backgrounds[1], mew=3, capsize=30, markersize=20)
            
            elif i==1 and j==1:
                errorbar(targets, db_exp_th4[:len(targets)], yerr=db_CI684[:,:len(targets)], fmt='ko', label=backgrounds[0], mew=3,  mfc='w', capsize=30, markersize=20)
                errorbar(targets, db_exp_th4[len(targets):], yerr=db_CI684[:,len(targets):], fmt='ko', label=backgrounds[1], mew=3, capsize=30, markersize=20)

            xlim([-0.5, 4.5])
            ylim([34, 66]) 
            ax.set_xticks([0,1,2,3,4])
            ax.set_xticks([0.5,1.5,2.5,3.5], minor=True)
            ax.set_yticks([34,38,42,46,50,54,58,62, 66])
            ax.set_yticks([36,40,44,48,52,56,60,64], minor=True)

            ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
            ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
            ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
            ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

            if j == 1:
                ax.tick_params(axis='y', labelleft=False)
            if i == 0:
                ax.tick_params(axis='x', labelbottom=False)

    #gcf().text(0.5, 0.02, 'target', ha='center',fontsize=36)
    gcf().text(0.001, 0.5, 'Amplitude threshold(dB)', va='center', rotation='vertical',fontsize=48)
    gcf().text(0.46, 0.001, 'Targets', va='center', rotation='horizontal',fontsize=48)
    tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)

    savefig(path+'/{}exp_th4{}.pdf'.format(theme, sessions4))
    savefig(path+'/{}exp_th4{}.png'.format(theme, sessions4))
    close()

#%%
def exp_th3(targets, backgrounds, sessions1, sessions2, sessions3, path='.', theme=''):
    db_exp_th1, db_CI681 = load_ths(path, sessions1)

    db_exp_th2, db_CI682 = load_ths(path, sessions2)

    db_exp_th3, db_CI683 = load_ths(path, sessions3)

    nrows = 1
    ncols = 3

    fig = figure(figsize=(32, 12))
    gs = fig.add_gridspec(nrows, ncols)
    colors = ['k', 'k', 'k']

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])

            if j==0:
                errorbar(targets, db_exp_th1[:len(targets)], yerr=db_CI681[:,:len(targets)], fmt='ko', label=backgrounds[0], mew=3,  mfc='w', capsize=30, markersize=20)
                errorbar(targets, db_exp_th1[len(targets):], yerr=db_CI681[:,len(targets):], fmt='ko', label=backgrounds[1], mew=3, capsize=30, markersize=20)

                legend(loc='upper left', fontsize=36)
            
            elif j==1:
                errorbar(targets, db_exp_th2[:len(targets)], yerr=db_CI682[:,:len(targets)], fmt='ko', label=backgrounds[0], mew=3,  mfc='w', capsize=30, markersize=20)
                errorbar(targets, db_exp_th2[len(targets):], yerr=db_CI682[:,len(targets):], fmt='ko', label=backgrounds[1], mew=3, capsize=30, markersize=20)

            elif j==2:
                errorbar(targets, db_exp_th3[:len(targets)], yerr=db_CI683[:,:len(targets)], fmt='ko', label=backgrounds[0], mew=3,  mfc='w', capsize=30, markersize=20)
                errorbar(targets, db_exp_th3[len(targets):], yerr=db_CI683[:,len(targets):], fmt='ko', label=backgrounds[1], mew=3, capsize=30, markersize=20)

            xlim([-0.5, 3.5])
            ylim([28, 48])
            ax.set_xticks([0,1,2,3])
            ax.set_xticks([0.5,1.5,2.5], minor=True)
            ax.set_yticks([28,30,32,34,36,38,40,42,44,46, 48])
            ax.set_yticks([29,31,33,35,37,39,41,43,45,47], minor=True)
            ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
            ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
            ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=False, labelright=True)
            ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

            if j==0:
                ax.tick_params(axis='y', which='major', labelleft=True)

    #gcf().text(0.5, 0.02, 'target', ha='center',fontsize=36)
    gcf().text(0.001, 0.5, 'Amplitude threshold(dB)', va='center', rotation='vertical',fontsize=48)
    gcf().text(0.45, 0.001, 'Targets', va='center', rotation='horizontal',fontsize=48)
    tight_layout(pad=2.5, h_pad=0.5, w_pad=0)

    savefig(path+'/{}exp_th3{}.pdf'.format(theme, sessions3))
    savefig(path+'/{}exp_th3{}.png'.format(theme, sessions3))
    close()

#%%
def fit_diff_cmp2_exp(targets, backgrounds, sessions, subject, models, cmp_th1, cmp_th2, path='.', theme=''):
    db_exp_th, db_CI68 = load_ths(path, sessions)

    nmods = len(models)
    nrows = nmods
    ncols = 2

    db_cmp_th = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th[0,0,:] = decibel(cmp_th1[7,:])
    db_cmp_th[0,1,:] = decibel(cmp_th1[3,:])
    db_cmp_th[1,0,:] = decibel(cmp_th2[7,:])
    db_cmp_th[1,1,:] = decibel(cmp_th2[3,:])

    scales = efficiency_log(targets, db_cmp_th, db_exp_th, 1, 2)
    rmse = np.zeros_like(scales)

    fig = figure(figsize=(32, 16))
    gs = fig.add_gridspec(nrows, ncols)
    colors = ['indigo', 'darkviolet']

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])

            if j==0:
                ax.errorbar(targets, db_exp_th[:len(targets)], yerr=db_CI68[:,:len(targets)], fmt='ko', mew=3,  mfc='w', capsize=30, markersize=20, zorder=0)
                ax.errorbar(targets, db_exp_th[len(targets):], yerr=db_CI68[:,len(targets):], fmt='ko', label=subject, mew=3, capsize=30, markersize=20, zorder=0)

                ax.scatter(targets, db_cmp_th[i,0,:]-scales[i], s=500, marker='D', edgecolor=colors[i],  facecolor="none", linewidths=3,  zorder=1)
                ax.scatter(targets, db_cmp_th[i,1,:]-scales[i], s=500, marker='D', color=colors[i], label=models[i], zorder=1)

                rmse[i] = RMSE_log(scales[i], db_exp_th[:len(targets)], db_exp_th[len(targets):], db_cmp_th[i,0,:], (db_cmp_th[i,1,:]))

                legend(loc='upper right', fontsize=36)

                # exp1
                # xlim([-0.5, 4.5])
                # ylim([34, 66]) 
                # ax.set_xticks([0,1,2,3,4])
                # ax.set_xticks([0.5,1.5,2.5,3.5], minor=True)
                # ax.set_yticks([34,38,42,46,50,54,58,62,66])
                # ax.set_yticks([36,40,44,48,52,56,60,64], minor=True)

                # exp2
                xlim([-0.5, 3.5])
                ylim([28, 48])
                ax.set_xticks([0,1,2,3])
                ax.set_xticks([0.5,1.5,2.5], minor=True)
                ax.set_yticks([28,32,36,40,44,48])
                ax.set_yticks([30,34,38,42,46], minor=True)

                ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
                ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
                ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
                ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

            else:
                scatter(targets, db_exp_th[:len(targets)]-db_cmp_th[i,0,:], s=500, marker='^', label=backgrounds[0], edgecolor='k',  facecolor="none", linewidths=3)
                scatter(targets, db_exp_th[len(targets):]-db_cmp_th[i,1,:], s=500, marker='^', color='k', label=backgrounds[1])

                if i == 0:
                    legend(loc='upper right', fontsize=36)

                ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
                ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
                ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
                ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

                xlim([-0.5, len(targets)-0.5])
                ylim([-6, 18])
                ax.set_xticks(np.arange(len(targets)))
                ax.set_xticks(np.arange(len(targets)-1)+0.5, minor=True)
                ax.set_yticks([-4,0,4,8,12,16])
                ax.set_yticks([-6,-2,2,6,10,14,18], minor=True)
                axhline(0, linestyle='dotted', linewidth=2, color='k', alpha=0.5)
            
            if i != 1:
                ax.tick_params(axis='x', labelbottom=False)

    gcf().text(0.002, 0.5, 'Adjusted amplitude threshold(dB)', va='center', rotation='vertical',fontsize=48)
    tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)
    gcf().text(0.46, 0.001, 'Targets', va='center', rotation='horizontal',fontsize=48)
    tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)
    savefig(path+'/{}fit_diff_cmp2_exp{}.pdf'.format(theme, sessions))
    savefig(path+'/{}fit_diff_cmp2_exp{}.png'.format(theme, sessions))
    close()

    return scales, rmse

#%%
def fit_diff_cmp3_exp(targets, backgrounds, sessions, subject, models, cmp_th1, cmp_th2, cmp_th3, path='.', theme=''):
    db_exp_th, db_CI68 = load_ths(path, sessions)

    nmods = len(models)
    nrows = nmods
    ncols = 2

    db_cmp_th = np.zeros((nmods,len(backgrounds),len(targets)))
    db_cmp_th[0,0,:] = decibel(cmp_th1[6,:])
    db_cmp_th[0,1,:] = decibel(cmp_th1[2,:])
    db_cmp_th[1,0,:] = decibel(cmp_th2[7,:])
    db_cmp_th[1,1,:] = decibel(cmp_th2[3,:])
    db_cmp_th[2,0,:] = decibel(cmp_th3[7,:])
    db_cmp_th[2,1,:] = decibel(cmp_th3[3,:])

    scales = efficiency_log(targets, db_cmp_th, db_exp_th, 1, 3)
    rmse = np.zeros_like(scales)

    fig = figure(figsize=(32, 24))
    gs = fig.add_gridspec(nrows, ncols)
    colors = ['forestgreen', 'indigo', 'darkviolet']

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i,j])

            if j==0:
                ax.errorbar(targets, db_exp_th[:len(targets)], yerr=db_CI68[:,:len(targets)], fmt='ko', mew=3,  mfc='w', capsize=30, markersize=20, zorder=0)
                ax.errorbar(targets, db_exp_th[len(targets):], yerr=db_CI68[:,len(targets):], fmt='ko', label=subject, mew=3, capsize=30, markersize=20, zorder=0)

                ax.scatter(targets, db_cmp_th[i,0,:]-scales[i], s=500, marker='D', edgecolor=colors[i],  facecolor="none", linewidths=3,  zorder=1)
                ax.scatter(targets, db_cmp_th[i,1,:]-scales[i], s=500, marker='D', color=colors[i], label=models[i], zorder=1)

                rmse[i] = RMSE_log(scales[i], db_exp_th[:len(targets)], db_exp_th[len(targets):], db_cmp_th[i,0,:], (db_cmp_th[i,1,:]))

                legend(loc='upper right', fontsize=36)

                # exp1
                # xlim([-0.5, 4.5])
                # ylim([34, 66]) 
                # ax.set_xticks([0,1,2,3,4])
                # ax.set_xticks([0.5,1.5,2.5,3.5], minor=True)
                # ax.set_yticks([34,38,42,46,50,54,58,62,66])
                # ax.set_yticks([36,40,44,48,52,56,60,64], minor=True)

                # exp2
                xlim([-0.5, 3.5])
                ylim([28, 48])
                ax.set_xticks([0,1,2,3])
                ax.set_xticks([0.5,1.5,2.5], minor=True)
                ax.set_yticks([28,32,36,40,44,48])
                ax.set_yticks([30,34,38,42,46], minor=True)

                ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
                ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
                ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
                ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

            else:
                scatter(targets, db_exp_th[:len(targets)]-db_cmp_th[i,0,:], s=500, marker='^', label=backgrounds[0], edgecolor='k',  facecolor="none", linewidths=3)
                scatter(targets, db_exp_th[len(targets):]-db_cmp_th[i,1,:], s=500, marker='^', color='k', label=backgrounds[1])

                if i == 0:
                    legend(loc='lower right', fontsize=36)

                ax.tick_params(axis='x', which='both', direction='out', length=0, width=0,pad =5, labelsize=36, labelbottom=True, labeltop=False, grid_color='k', grid_alpha=1, grid_linewidth=1, grid_linestyle='--')
                ax.grid(b=True, which='minor', axis='x', linestyle='--', linewidth=2)
                ax.tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=36, left=True, right=True, labelleft=True, labelright=True)
                ax.tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=True, labelleft=False, labelright=False)

                # exp1
                # xlim([-0.5, 4.5])
                # ylim([-2, 10])
                # ax.set_xticks([0,1,2,3,4])
                # ax.set_xticks([0.5,1.5,2.5,3.5], minor=True)
                # ax.set_yticks([-2,0,2,4,6,8,10])
                # ax.set_yticks([-1,1,3,5,7,9], minor=True)

                # exp2
                xlim([-0.5, 3.5])
                ylim([-2, 10])
                ax.set_xticks([0,1,2,3])
                ax.set_xticks([0.5,1.5,2.5], minor=True)
                ax.set_yticks([-2,0,2,4,6,8,10])
                ax.set_yticks([-1,1,3,5,7,9], minor=True)
            
            if i != 2:
                ax.tick_params(axis='x', labelbottom=False)

    gcf().text(0.002, 0.5, 'Adjusted amplitude threshold(dB)', va='center', rotation='vertical',fontsize=48)
    tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)
    gcf().text(0.46, 0.001, 'Targets', va='center', rotation='horizontal',fontsize=48)
    tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)
    savefig(path+'/{}fit_diff_cmp3_exp{}.pdf'.format(theme, sessions))
    savefig(path+'/{}fit_diff_cmp3_exp{}.png'.format(theme, sessions))
    close()

    return scales, rmse

#%%
def delta_cmps_th(backgrounds, models, colors, delta_thresholds, path, theme=''):
    figure(figsize=(12, 8))
    x = np.arange(len(models))
    bar(x, delta_thresholds[0], color=colors[0], label = backgrounds[0], width = 0.25)
    bar(x + 0.25, delta_thresholds[1], color=colors[1], label = backgrounds[1], width = 0.25)
    legend(loc='upper right', prop={'size':24})
    rc('text', usetex=True)
    rcParams["text.usetex"] = True
    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    xlabel(r'\textbf{Model\,\,observer}', fontsize = 36)
    ylabel(r'$\textbf{Th}_\textbf{TM}-\textbf{Th}_\textbf{M}\,\textbf{(dB)}$', fontsize = 36)
    xticks(x+0.125, models)
    ymax = 10
    ylim([0, ymax])

    gca().set_yticks(np.linspace(0,ymax,ymax+1))
    gca().set_yticks(np.linspace(0.5,ymax-0.5,ymax), minor=True)
    gca().tick_params(axis='x', which='both', direction='out', length=0, width=0, pad =5, labelsize=36, labelbottom=True, labeltop=False)
    gca().tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=30, left=True, right=False, labelleft=True, labelright=False)
    gca().tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=False, labelleft=False, labelright=False)

    savefig(path+'/{}delta_cmps_th.pdf'.format(theme))
    savefig(path+'/{}delta_cmps_th.png'.format(theme))
    close()

#%%
def rank_rms_scale(models, rms, scale, path, theme=''):
    colors = ['k'] * len(models)

    fig = figure(figsize=(32, 16))
    gs = fig.add_gridspec(1, 2)
    order = np.flip(np.argsort(rms))
    ordered_models = [models[i] for i in order]
    x = np.arange(len(models))
    
    ax0 = fig.add_subplot(gs[0,0])
    bar(x, rms[order], color=colors[0], width = 0.5)
    xlabel('Model observer', fontweight ='bold', fontsize = 36)
    ylabel('RMS error (dB)', fontweight ='bold', fontsize = 36)
    xticks(x, ordered_models)

    gca().set_yticks(np.linspace(0,5,6))
    gca().set_yticks(np.linspace(0.5,4.5,5), minor=True)
    gca().tick_params(axis='x', which='both', direction='out', length=0, width=0, pad =5, labelsize=36, labelbottom=True, labeltop=False)
    gca().tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=30, left=True, right=False, labelleft=True, labelright=False)
    gca().tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=False, labelleft=False, labelright=False)

    ax1 = fig.add_subplot(gs[0,1])
    bar(x, scale[order], color=colors[0], width = 0.5)
    xlabel('Model observer', fontweight ='bold', fontsize = 36)
    ylabel('Efficiency scale factor', fontweight ='bold', fontsize = 36)
    xticks(x, ordered_models)

    gca().set_yticks(np.linspace(0,1,6))
    gca().set_yticks(np.linspace(0.1,0.9,5), minor=True)
    gca().tick_params(axis='x', which='both', direction='out', length=0, width=0, pad =5, labelsize=36, labelbottom=True, labeltop=False)
    gca().tick_params(axis='y', which='major', direction='out', length=12, width=4, pad = 3, labelsize=30, left=True, right=False, labelleft=True, labelright=False)
    gca().tick_params(axis='y', which='minor', direction='out', length=8, width=4, left=True, right=False, labelleft=False, labelright=False)

    savefig(path+'/{}rank_rms_scale.pdf'.format(theme))
    savefig(path+'/{}rank_rms_scale.png'.format(theme))
    close()

#%%
def akaike_information_criterion(cmp_th, targets, models, n_params, sessions, path):
    exp_ths = np.load(path+'/bs_ths{}.npy'.format(sessions))

    nmods = len(models)
    ntars = len(targets)
    
    uni_cmp_th = cmp_th[nmods:,:]
    mod_cmp_th = cmp_th[:nmods,:]

    nmods = len(models)

    db_cmp_th = np.zeros((nmods,2,len(targets)))
    db_cmp_th[:,0,:] = decibel(uni_cmp_th)
    db_cmp_th[:,1,:] = decibel(mod_cmp_th)

    db_exp_th, db_CI68 = load_ths(path, sessions)
    db_scales = efficiency_log(targets, db_cmp_th, db_exp_th, nmods//2, nmods//2)
    scales = 10**(db_scales / 20)
    
    ll = np.zeros((nmods,))
    for j in range(ntars):
        uni_exp_ths = exp_ths[:,j]
        mod_exp_ths = exp_ths[:,ntars+j]
        
        hist(uni_exp_ths)
        hist(mod_exp_ths)
        savefig(f't_{j}.png')
        
        for i in range(nmods):
            ll[i] += np.log(norm.pdf(uni_cmp_th[i,j]/scales[i], uni_exp_ths.mean(), uni_exp_ths.std()))
            ll[i] += np.log(norm.pdf(mod_cmp_th[i,j]/scales[i], mod_exp_ths.mean(), mod_exp_ths.std()))

    return 2 * ( n_params - ll)
