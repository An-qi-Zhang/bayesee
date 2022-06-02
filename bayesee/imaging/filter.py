import numpy as np
from numba import njit, prange
import math
import scipy as sp

#%%
def filter_fft(x, fft_filter):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(x))*fft_filter)))

#%%
def luminance_modulate_straight(x, direction, cut, ratio, flip=True):
    if direction == 'h':
        x_mod = np.hstack((x[:,:cut]/np.sqrt(ratio),x[:,cut:]*np.sqrt(ratio)))
        if flip: x_mod = np.fliplr(x_mod)
    elif direction == 'v':
        x_mod = np.vstack((x[:cut,:]/np.sqrt(ratio),x[cut:,:]*np.sqrt(ratio)))
        if flip: x_mod = np.flipud(x_mod) 
    else:
        raise ValueError('Direction is undefined.')
    
    return x_mod

#%%
def luminance_modulate_lr(x, ratio, flip=False):
    return luminance_modulate_straight(x, 'h', round(x.shape[1]/2), ratio, flip)

#%%
def luminance_modulate_ud(x, ratio, flip=False):
    return luminance_modulate_straight(x, 'v', round(x.shape[0]/2), ratio, flip)

#%%
def contrast_modulate_straight(x, direction, cut, ratio, flip=False):
    row, col = x.shape
    x_mean = x.mean()
    x -= x_mean
    if direction == 'h':
        x_mod = np.hstack((x[:,:cut]/np.sqrt(ratio),x[:,cut:]*np.sqrt(ratio)))
        if flip: x_mod = np.fliplr(x_mod)
        p = (col - cut - 1) / col
    elif direction == 'v':
        x_mod = np.vstack((x[:cut,:]/np.sqrt(ratio),x[cut:,:]*np.sqrt(ratio)))
        if flip: x_mod = np.flipud(x_mod) 
        p = (row - cut - 1) / row
    else:
        raise ValueError('Direction is undefined.')
    
    x_mod *= np.sqrt((p**2+(1-p)**2)/(p**2/ratio+(1-p)**2*ratio)) # keep the overall contrast
    x_mod = x_mod - x_mod.mean() + x_mean
    return x_mod

#%%
def contrast_modulate_lr(x, ratio, flip=False):
    return contrast_modulate_straight(x, 'h', round(x.shape[1]/2), ratio, flip)

#%%
def contrast_modulate_ud(x, ratio, flip=False):
    return contrast_modulate_straight(x, 'v', round(x.shape[0]/2), ratio, flip)
    
#%%
def exponential_distance(row, col, v_center, h_center, exponent):
    xx, yy = np.meshgrid(range(row), range(col), sparse=True, indexing='ij') # not Cartesian
    if exponent >= 0:
        return np.hypot(xx - v_center, yy - h_center)**exponent
    else:
        distance = np.hypot(xx - v_center, yy - h_center)
        distance[distance!=0] = distance[distance!=0]**exponent
        return distance

#%%
def distance_to_point(row, col, v_center, h_center):
    return exponential_distance(row, col, v_center, h_center, 1)

#%%
def circ_mask(row, col, radius):
    mask = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if (i-row/2)**2+(j-col/2)**2 <= radius**2:
                mask[i,j]=1
    
    return mask

#%%
def square_mask(row, col, m_row, m_col):
    mask = np.zeros((row, col))
    mask[(row-m_row)//2:(row+m_row)//2, (col-m_col)//2:(col+m_col)//2]=1
    
    return mask

#%%
def local_std(x, size):
    return sp.ndimage.generic_filter(x, np.std, size)

#%%
def csf_filter(row, col, ppd=60, d=4, w=550, a=0.85, b=0.15, c=0.065, n=2):
    # a: surround strength
    # b: surround size (mm)
    # c: center size (mm)
    # n: surround shape exponent
    # d: pupil diameter (mm)
    # w: wavelength (nm)
    # freq_deg: spatial frequency (cycles/degree)
    
    xx, yy = np.meshgrid(range(row), range(col), sparse=True)
    xx_img = xx - (row-row%2)/2
    yy_img = yy - (col-col%2)/2
    xx_deg = xx_img / (row/ppd)
    yy_deg = yy_img / (col/ppd)
    freq_deg = np.hypot(xx_deg, yy_deg)
    
    u0 = (d*np.pi()*10**6/(w*180))
    u1 = 21.95 - 5.512*d + 0.3922*d**2
    uh = freq_deg/ u0
    D = (np.arccos(uh) - uh*np.sqrt(1-uh**2))*2/np.pi
    otf = np.sqrt(D)*(1 + (freq_deg/u1)**2)**(-0.62)
    csf = otf**(1-a*np.exp(-b*freq_deg**n))**np.exp(-c*freq_deg)
    
    return csf

#%%
@njit(cache = True, fastmath=True, parallel=True, nogil=True)
def sp_Drasdo(row, col, ppd):
    e_yp2 = (1.13 * ppd)**2
    e_yn2 = (1.49 * ppd)**2
    e_xp2 = (1.67 * ppd)**2
    e_xn2 = (1.63 * ppd)**2
    s0 = 1.0/120 * ppd

    sp = np.zeros((row, col))
    for i in prange(row):
        for j in prange(col):
            if i>=row//2:
                if j>=col//2:
                    sp[i,j] = s0 * (math.sqrt((i-row//2)**2/e_yp2+(j-col//2)**2/e_xp2)+1)
                else:
                    sp[i,j] = s0 * (math.sqrt((i-row//2)**2/e_yp2+(j-col//2)**2/e_xn2)+1)
            else:
                if j>=col//2:
                    sp[i,j] = s0 * (math.sqrt((i-row//2)**2/e_yn2+(j-col//2)**2/e_xp2)+1)
                else:
                    sp[i,j] = s0 * (math.sqrt((i-row//2)**2/e_yn2+(j-col//2)**2/e_xn2)+1)

    return sp

#%%
def dp_Drasdo(row, col, ppd, eta, dp0, phi):
    # phi in degree^-1
    sp = sp_Drasdo(row, col, ppd)
    dp = eta * dp0 * (1+phi/ppd*sp[row//2,col//2]) / (1+phi/ppd*sp)

    return dp

#%%