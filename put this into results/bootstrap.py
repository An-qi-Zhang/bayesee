#%%
import numpy as np
from datetime import datetime
import scipy.optimize as opt
import bayesfit as bf
from numba import njit, prange
from math import erf, sqrt, log10

#%%
@njit(cache = True, fastmath=True, nogil=True)
def neg_log_likelihood(x, data):
    if x[0] < 0:
        return np.Inf
    neg_ll = 0
    for i in prange(data.shape[0]):
        neg_ll -= data[i,1]*log10(0.5*(erf(1/sqrt(2)*(0.5*(data[i,0]/x[0])**x[1]-x[2]))+1))
        neg_ll -= (data[i,3]/2-data[i,1])*log10(1-0.5*(erf(1/sqrt(2)*(0.5*(data[i,0]/x[0])**x[1]-x[2]))+1))
        neg_ll -= data[i,2]*log10(0.5*(erf(1/sqrt(2)*(-0.5*(data[i,0]/x[0])**x[1]-x[2]))+1))
        neg_ll -= (data[i,3]/2-data[i,2])*log10(1-0.5*(erf(1/sqrt(2)*(-0.5*(data[i,0]/x[0])**x[1]-x[2]))+1))
    
    return neg_ll

#%%
def across_levels(amps, acc, tar_pre, subject, n_samp=1000, path_save='.'):
    
    n_params = amps.shape[1]
    slopes_MC = np.zeros((n_samp, n_params))
    shapes_MC = np.zeros((n_samp, n_params))
    criteria_MC = np.zeros((n_samp, n_params))
    ths_MC = np.zeros((n_samp, n_params))

    acc = np.asarray(acc, dtype=int)
    data = np.zeros((amps.shape[2],4)) # amplitude, hit, false alarm, total
    data[:,3] = acc.shape[0]*amps.shape[0] # nTrials * nSessions

    init_slopes = np.ones((n_params,1)) * 500
    init_shapes = np.ones((n_params,1)) * 1.25
    init_criteria = np.zeros((n_params,1))

    for i in range(n_samp):
        for j in range(n_params):

                index_amp = np.random.choice(amps.shape[2], amps.shape[2])
                # amplitude
                data[:,0] = amps[0,j, index_amp]
                # hit rate
                data[:,1] = (tar_pre[:,:,j,index_amp]*acc[:,:,j,index_amp]).sum(axis=(0,1))
                # false alarm rate
                data[:,2] = ((1-tar_pre[:,:,j,index_amp])*(1-acc[:,:,j,index_amp])).sum(axis=(0,1))
                
                x_opt = opt.minimize(neg_log_likelihood, (init_slopes[j],init_shapes[j],init_criteria[j]), args=(data), method='SLSQP', bounds=((amps[:,j,:].min(),amps[:,j,:].max()), (0.5,5), (-50, 50))).x

                slopes_MC[i,j] = x_opt[0]
                shapes_MC[i,j] = x_opt[1]
                criteria_MC[i,j] = x_opt[2]
                ths_MC[i,j] = x_opt[0]
                
        print("progress:", (i+1) * 100.0 / n_samp, "%")

    np.save(path_save+'/bs_slopes_{}'.format(subject), slopes_MC)
    np.save(path_save+'/bs_shapes_{}'.format(subject), shapes_MC)
    np.save(path_save+'/bs_criteria_{}'.format(subject), criteria_MC)
    np.save(path_save+'/bs_ths_{}'.format(subject), ths_MC)
    
    end_time = datetime.now()
    print('start time = ')
    show_time(start_time)
    print('end time = ')
    show_time(end_time)
    print('time cost = \n\t\t', end_time - start_time, '(h/m/s)')

#%%
def across_bin_trials(amps, acc, tar_pre, subject, n_bin, n_bins, n_samp=1000, path_save='.'):
    
    n_params = amps.shape[1]
    slopes_MC = np.zeros((n_samp, n_params))
    shapes_MC = np.zeros((n_samp, n_params))
    criteria_MC = np.zeros((n_samp, n_params))
    ths_MC = np.zeros((n_samp, n_params))

    acc = np.asarray(acc, dtype=int)
    data = np.zeros((amps.shape[0],4)) # amplitude, hit, false alarm, total
    data[:,3] = 1

    init_slopes = np.ones((n_params,1)) * 270
    init_shapes = np.ones((n_params,1)) * 1
    init_criteria = np.zeros((n_params,1))

    for i in range(n_samp):
        for j in range(n_params):
                index_amp = np.random.choice(amps.shape[0], amps.shape[0])
                # amplitude
                data[:,0] = amps[index_amp, j]
                # hit rate
                data[:,1] = (tar_pre[index_amp, j]*acc[index_amp, j])
                # false alarm rate
                data[:,2] = ((1-tar_pre[index_amp, j])*(1-acc[index_amp, j]))
                
                x_opt = opt.minimize(neg_log_likelihood, (init_slopes[j],init_shapes[j],init_criteria[j]), args=(data), method='SLSQP', bounds=((amps[:,j].min(),amps[:,j].max()*2), (0.5,5), (-50, 50))).x

                slopes_MC[i,j] = x_opt[0]
                shapes_MC[i,j] = x_opt[1]
                criteria_MC[i,j] = x_opt[2]
                ths_MC[i,j] = x_opt[0]
                
        print("progress:", (i+1) * 100.0 / n_samp, "%")

    np.save(path_save+'/bs_slopes_{}_{}of{}'.format(subject, n_bin, n_bins), slopes_MC)
    np.save(path_save+'/bs_shapes_{}_{}of{}'.format(subject, n_bin, n_bins), shapes_MC)
    np.save(path_save+'/bs_criteria_{}_{}of{}'.format(subject, n_bin, n_bins), criteria_MC)
    np.save(path_save+'/bs_ths_{}_{}of{}'.format(subject, n_bin, n_bins), ths_MC)
    
    end_time = datetime.now()
    print('start time = ')
    show_time(start_time)
    print('end time = ')
    show_time(end_time)
    print('time cost = \n\t\t', end_time - start_time, '(h/m/s)')

#%%
@njit(cache = True, fastmath=True, nogil=True)
def neg_log_likelihood(x, data):
    if x[0] < 0:
        return np.Inf
    neg_ll = 0
    for i in prange(data.shape[0]):
        neg_ll -= data[i,1]*log10(0.5*(erf(1/sqrt(2)*(0.5*(data[i,0]/x[0])**x[1]-x[2]))+1))
        neg_ll -= (data[i,3]/2-data[i,1])*log10(1-0.5*(erf(1/sqrt(2)*(0.5*(data[i,0]/x[0])**x[1]-x[2]))+1))
        neg_ll -= data[i,2]*log10(0.5*(erf(1/sqrt(2)*(-0.5*(data[i,0]/x[0])**x[1]-x[2]))+1))
        neg_ll -= (data[i,3]/2-data[i,2])*log10(1-0.5*(erf(1/sqrt(2)*(-0.5*(data[i,0]/x[0])**x[1]-x[2]))+1))
    
    return neg_ll

#%%
def across_level(sessions, amps, acc, tar_pre, n_samp=100, path_save='.'):
    start_time = datetime.now()
    
    n_params = amps.shape[0]//2*amps.shape[1]
    slopes_MC = np.zeros((n_samp, n_params))
    shapes_MC = np.zeros((n_samp, n_params))
    criteria_MC = np.zeros((n_samp, n_params))
    ths_MC = np.zeros((n_samp, n_params))

    acc = np.asarray(acc, dtype=int)
    data = np.zeros((acc.shape[3]*2,4))
    data[:,3] = acc.shape[0]

    init_slopes = np.ones((acc.shape[1]//2, acc.shape[2]))
    init_shapes = np.ones((acc.shape[1]//2, acc.shape[2])) * 3
    init_criteria = np.zeros((acc.shape[1]//2, acc.shape[2]))

    #exp1
    # init_slopes[0,0] = 600
    # init_slopes[0,1:4] = 400
    # init_slopes[0,4] = 200
    # init_slopes[1,0:3] = 300
    # init_slopes[1,3:5] = 200
    # init_shapes[:,:] = 3

    #exp2
    init_slopes[0,:] = 100
    init_slopes[1,:] = 50
    init_shapes[0,:] = 1
    init_shapes[1,:] = 1.25

    for i in range(n_samp):
        for j in range(acc.shape[1]//2):
            for k in range(acc.shape[2]):

                index_amp = np.random.choice(acc.shape[3], acc.shape[3])
                data[:acc.shape[3],0] = amps[j, k, index_amp]
                #data[:,1] hit rate
                data[:acc.shape[3],1] = (tar_pre[:,j,k,index_amp]*acc[:,j,k,index_amp]).sum(axis=0)
                #data[:,2] false alarm rate
                data[:acc.shape[3],2] = ((1-tar_pre[:,j,k,index_amp])*(1-acc[:,j,k,index_amp])).sum(axis=0)

                index_amp = np.random.choice(acc.shape[3], acc.shape[3])
                data[acc.shape[3]:,1] = (tar_pre[:,j,k,index_amp]*acc[:,3-j,k,index_amp]).sum(axis=0)
                data[acc.shape[3]:,2] = ((1-tar_pre[:,3-j,k,index_amp])*(1-acc[:,3-j,k,index_amp])).sum(axis=0)
                
                x_opt = opt.minimize(neg_log_likelihood, (init_slopes[j,k],init_shapes[j,k],init_criteria[j,k]), args=(data), method='SLSQP', bounds=((0.001,amps[:,k,:].max()), (0.5,10), (-2, 5))).x

                slopes_MC[i,j*acc.shape[2]+k] = x_opt[0]

                shapes_MC[i,j*acc.shape[2]+k] = x_opt[1]

                criteria_MC[i,j*acc.shape[2]+k] = x_opt[2]

                ths_MC[i,j*acc.shape[2]+k] = x_opt[0]*(2*x_opt[2]+1.0)**(1.0/x_opt[1])
                

        print("progress:", (i+1) * 100.0 / n_samp, "%")

    np.save(path_save+'/bs_slopes{}'.format(sessions), slopes_MC)

    np.save(path_save+'/bs_shapes{}'.format(sessions), shapes_MC)

    np.save(path_save+'/bs_criteria{}'.format(sessions), criteria_MC)

    np.save(path_save+'/bs_ths{}'.format(sessions), ths_MC)

#%%
