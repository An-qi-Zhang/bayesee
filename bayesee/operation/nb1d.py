#%% Numba is only worthy when the operation decreases data dimension/extract statistic summary.

from numba import njit, prange
import math

#%%
@njit(cache = True, fastmath=True, parallel=True, nogil=True)
def nb1mean(x):
    return x.mean()

#%%
@njit(cache = True, fastmath=True, parallel=True, nogil=True)
def nb1std(x):
    return x.std()

#%%
@njit(cache = True, fastmath=True, parallel=True, nogil=True)
def nb1norm(x):
    norm_x = 0
    for i in prange(len(x)):
            norm_x += x[i]**2

    return math.sqrt(norm_x)

@njit(cache = True, fastmath=True, parallel=True, nogil=True)
def nb1sum(x):
    sum_x = 0
    for i in prange(x.shape[0]):
            sum_x += x[i]

    return sum_x
