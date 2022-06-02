#%%
import numpy as np
from scipy.stats import norm, t
import math

#%%
def neg_ll_2rho(x, data):
    # Reference: Sebastian, Stephen, and Wilson S. Geisler. "Decision-variable correlation." Journal of Vision 18.4 (2018): 3-3.
    
    # x[0] = rho_a
    # x[1] = rho_b
    
    # data[:,0] = amp
    # data[:,1] = 0-a 1-b (stimulus)
    # data[:,2] = qa / qb (response 0-A 1-B)
    # data[:,3] = zma / zmb (grouped z-score)
    # data[:,4] = alpha
    # data[:,5] = beta
    # data[:,6] = gamma_s

    if x[0] < -1 or x[0] > 1 or x[1] < -1 or x[1] > 1:
        return 1e10

    neg_ll = 0
    
    dps = np.log((np.exp(data[:,0]*data[:,4])+data[:,5])/(1+data[:,5]))
    
    a = data[:,1] == 0
    pAa = norm.cdf((data[a,6]+0.5*dps[a]-x[0]*data[a,3])/math.sqrt(1-x[0]**2))
    b = data[:,1] == 1
    pAb = norm.cdf((data[b,6]-0.5*dps[b]-x[1]*data[b,3])/math.sqrt(1-x[1]**2))
    
    neg_ll -= np.sum((1-data[a,2])*np.log(pAa)+data[a,2]*np.log(1-pAa))
    neg_ll -= np.sum((1-data[b,2])*np.log(pAb)+data[b,2]*np.log(1-pAb))
    
    return neg_ll

#%%
def neg_ll_1rho(x, data):
    # based on the reference: Sebastian, Stephen, and Wilson S. Geisler. "Decision-variable correlation." Journal of Vision 18.4 (2018): 3-3.
    
    # x = rho
    
    # data[:,0] = amp
    # data[:,1] = 0-a 1-b (stimulus)
    # data[:,2] = qa / qb (response)
    # data[:,3] = zma / zmb
    # data[:,4] = alpha
    # data[:,5] = beta
    # data[:,6] = gamma_s

    if x < -1 or x > 1:
        return 1e10

    neg_ll = 0
    
    dps = np.log((np.exp(data[:,0]*data[:,4])+data[:,5])/(1+data[:,5]))
    
    a = data[:,1] == 0
    pAa = norm.cdf((data[a,6]+0.5*dps[a]-x*data[a,3])/math.sqrt(1-x**2))
    b = data[:,1] == 1
    pAb = norm.cdf((data[b,6]-0.5*dps[b]-x*data[b,3])/math.sqrt(1-x**2))
    
    neg_ll -= np.sum((1-data[a,2])*np.log(pAa)+data[a,2]*np.log(1-pAa))
    neg_ll -= np.sum((1-data[b,2])*np.log(pAb)+data[b,2]*np.log(1-pAb))
    
    return neg_ll