import numpy as np
from scipy.stats import norm, t
import math

#%%
def t_CI_width(a, confidence, axis=0):
    return np.std(a, axis=axis) * t.ppf((1 + confidence) / 2., len(a)-1)
