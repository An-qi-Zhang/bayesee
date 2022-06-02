#%% Numba is only worthy when the operation decreases data dimension/extract statistic summary.

from numba import njit, prange
import numpy as np
import math

#%%
@njit(cache = True, fastmath=True, parallel=True, nogil=True)
def nb2mean(x):
    return x.mean()

#%%
@njit(cache = True, fastmath=True, parallel=True, nogil=True)
def nb2std(x):
    return x.std()

#%%
@njit(cache = True, fastmath=True, parallel=True, nogil=True)
def nb2norm(x):
    norm_x = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            norm_x += x[i,j]**2

    return math.sqrt(norm_x)

#%%
@njit(cache = True, fastmath=True, parallel=True, nogil=True)
def nb2sum(x):
    sum_x = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            sum_x += x[i,j]

    return sum_x

#%%
@njit(cache = True, fastmath=True, parallel=True, nogil=True)
def nb2dot(x, y):
    dot = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            dot += x[i,j] * y[i,j]

    return dot

#%%
@njit(cache = True, fastmath=True, nogil=True)
def nb2cross(x, y):
    xr, xc = x.shape
    yr, yc = y.shape
    cc = np.zeros((xr, xc))
    if xr>=yr or xc>=yc:
        patch = np.zeros((yr,yc))
        for i in prange(xr):
            for j in prange(xc):
                yrt = np.array((0, yr//2-i)).max()
                yrb = np.array((yr, yr//2+(xr-i))).min()
                ycl = np.array((0, yc//2-j)).max()
                ycr = np.array((yc, yc//2+(xc-j))).min()
                xrt = np.array((0, i-yr//2)).max()
                xrb = np.array((xr, i+yr//2)).min()
                xcl = np.array((0, j-yc//2)).max()
                xcr = np.array((xc, j+yc//2)).min()

                patch[yrt:yrb,ycl:ycr] = x[xrt:xrb,xcl:xcr]

                cc[i,j] = (y*patch).sum()
    else:
        for i in prange(xr):
            for j in prange(xc):
                cc[i,j] = (x*y[int(yr/2-i)-1:int(yr/2+(xr-i)-1), int(yc/2-j)-1:int(yc/2+(xc-j))-1]).sum()
    return cc

#%%
@njit(cache = True, fastmath=True, nogil=True)
def nb2RMSE(x, y):
    sum = nb2sum((x-y)**2)
    return math.sqrt(sum)