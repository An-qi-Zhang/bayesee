#%%
import numpy as np
from scipy.stats import norm, t
from scipy.optimize import curve_fit

#%%
def glm_disc(amp, stimulus, response):
    # amp with values that are same or different or hybrid
    # stimulus: 0-a 1-b, 1d array
    # response: discrete response 0-A 1-B, 1d array
    
    dp = np.zeros_like(amp)
    gamma = np.zeros_like(amp)
    unq_amps = np.unique(amp)
    
    for i, unq_amp in enumerate(unq_amps):
        id = amp == unq_amp
        pBb = np.sum((stimulus==1) & (response==1) & id)/len(id)
        pBa = np.sum((stimulus==0) & (response==1) & id)/len(id)
        dp[id] = norm.ppf(pBb) - norm.ppf(pBa)
        gamma[id] = -0.5 * (norm.ppf(pBb) + norm.ppf(pBa))
    
    return dp, gamma

#%%
def glm_cont(amp, stimulus, response):
    # amp with values that are same or different or hybrid
    # stimulus: 0-a 1-b
    # response: continuous response
    
    dp = np.zeros_like(amp)
    gamma = np.zeros_like(amp)
    unq_amps = np.unique(amp)
    
    for i, unq_amp in enumerate(unq_amps):
        id = amp == unq_amp
        ra, rb = response[(stimulus==0) & id], response[(stimulus==1) & id]
        na, nb = len(ra), len(rb)
        ma, mb = ra.mean(), rb.mean()
        va, vb = ra.var(), rb.var()
        sd_ave = np.sqrt((na*va+nb*vb)/(na+nb-2))
        dp[id] = abs(mb - ma)/sd_ave
        gamma[id] = (na*vb*ma+nb*va*mb)/(na*vb+nb*va)
    
    return dp, gamma

#%%
def linear_dp_amp(amp, k):
    return k*amp

#%%
def uncertain_dp_amp(amp, alpha, beta):
    # Reference: Sebastian, Stephen, and Wilson S. Geisler. "Decision-variable correlation." Journal of Vision 18.4 (2018): 3-3.
    return np.log((np.exp(amp*alpha)+beta)/(1+beta))

#%%
def glm_disc_linear(amp, stimulus, response):
    # amp with values that are same or different or hybrid
    # stimulus: 0-a 1-b, 1d array
    # response: discrete response 0-A 1-B, 1d array
    
    dp, gamma = glm_disc(amp, stimulus, response)
    k,_ = curve_fit(linear_dp_amp, amp, dp, bounds=((0,np.inf)))
    return dp, gamma, k

#%%
def glm_cont_linear(amp, stimulus, response):
    # amp with values that are same or different or hybrid
    # stimulus: 0-a 1-b, 1d array
    # response: discrete response 0-A 1-B, 1d array
    
    dp, gamma = glm_cont(amp, stimulus, response)
    k,_ = curve_fit(linear_dp_amp, amp, dp, bounds=((0,np.inf)))
    return dp, gamma, k

#%%
def glm_disc_uncertainty(amp, stimulus, response):
    # amp with values that are same or different or hybrid
    # stimulus: 0-a 1-b, 1d array
    # response: discrete response 0-A 1-B, 1d array
    
    dp, gamma = glm_disc(amp, stimulus, response)
    (alpha, beta),_ = curve_fit(uncertain_dp_amp, amp, dp, bounds=((0,np.inf), (0,np.inf)))
    return dp, gamma, alpha, beta

#%%
def glm_cont_uncertainty(amp, stimulus, response):
    # amp with values that are same or different or hybrid
    # stimulus: 0-a 1-b
    # response: continuous response
    
    dp, gamma = glm_cont(amp, stimulus, response)
    (alpha, beta),_ = curve_fit(uncertain_dp_amp, amp, dp, bounds=((0,np.inf), (0,np.inf)))
    return dp, gamma, alpha, beta

#%%
def linear_disc_th(amp, stimulus, response):
    # dp = 1
    dp, gamma, k = glm_disc_linear(amp, stimulus, response)
    return k

#%%
def linear_cont_th(amp, stimulus, response):
    # dp = 1
    dp, gamma, k = glm_cont_linear(amp, stimulus, response)
    return k

#%%
def uncertain_disc_th(amp, stimulus, response):
    # dp = 1
    alpha, beta = glm_disc_uncertainty(amp, stimulus, response)
    return np.log(np.e*(1+beta)-beta)/alpha

#%%
def uncertain_cont_th(amp, stimulus, response):
    # dp = 1
    alpha, beta = glm_cont_uncertainty(amp, stimulus, response)
    return np.log(np.e*(1+beta)-beta)/alpha