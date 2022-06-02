#%%
import numpy as np
from skimage.transform import resize
from imaging.filter import *
from scipy.signal import sawtooth, square

#%%
def add_by_coord(large, small, upleft_i, upleft_j):
    # 2D image: small or equal size for the first variable
    # upleft_i, upleft_j need to be integers
    s_row, s_col = small.shape
    l_row, l_col = large.shape

    added = np.ones_like(large)
    added[upleft_i:upleft_i+s_row, upleft_j:upleft_j+s_col] += small

    return added

#%%
def add_to_center(large, small):
    # 2D image: small or equal size for the first variable
    s_row, s_col = small.shape
    l_row, l_col = large.shape
    
    if (l_row-s_row) % 2:
        s_row += 1
    
    if (l_col-s_col) % 2:
        s_col += 1
        
    small = resize(small, (s_row, s_col))
    
    upleft_i = int((l_row-s_row) / 2)
    upleft_j = int((l_col-s_col) / 2)
    
    return add_by_coord(large, small, upleft_i, upleft_j)

#%%
def occlude_by_coord(large, small, upleft_i, upleft_j):
    # 2D image: small or equal size for the first variable
    # upleft_i, upleft_j need to be integers
    s_row, s_col = small.shape
    l_row, l_col = large.shape

    occluded = np.ones_like(large)
    occluded[upleft_i:upleft_i+s_row, upleft_j:upleft_j+s_col] = small

    return occluded

#%%
def occlude_center(large, small):
    # 2D image: small or equal size for the first variable
    s_row, s_col = small.shape
    l_row, l_col = large.shape
    
    if (l_row-s_row) % 2:
        s_row += 1
    
    if (l_col-s_col) % 2:
        s_col += 1
        
    small = resize(small, (s_row, s_col))
    
    upleft_i = int((l_row-s_row) / 2)
    upleft_j = int((l_col-s_col) / 2)
    
    return occlude_by_coord(large, small, upleft_i, upleft_j)

#%%
def cut_by_coord(large, small, upleft_i, upleft_j):
    # 2D image: small or equal size for the first variable
    # upleft_i, upleft_j need to be integers
    s_row, s_col = small.shape

    return large[upleft_i:upleft_i+s_row, upleft_j:upleft_j+s_col]

#%%
def cut_center(large, small):
    # 2D image: small or equal size for the first variable
    s_row, s_col = small.shape
    l_row, l_col = large.shape
    
    if (l_row-s_row) % 2:
        s_row += 1
    
    if (l_col-s_col) % 2:
        s_col += 1
        
    small = resize(small, (s_row, s_col))
    
    upleft_i = int((l_row-s_row) / 2)
    upleft_j = int((l_col-s_col) / 2)
    
    return cut_by_coord(large, small, upleft_i, upleft_j)

#%%
def cut_randomly(large, small):
    # 2D image: small or equal size for the first variable
    s_row, s_col = small.shape
    l_row, l_col = large.shape
    
    upleft_i = np.random.randint(l_row-s_row)
    upleft_j = np.random.randint(l_col-s_col)

    return large[upleft_i:upleft_i+s_row, upleft_j:upleft_j+s_col]

#%%
def power_noise(row, col, power, mean=0, std=1):
    white_noise = np.random.normal(size=(row, col))
    power_filter = exponential_distance(row, col, (row-row%2)/2, (col-col%2)/2, power)
    empowered_noise = filter_fft(white_noise, power_filter)
    empowered_noise -= empowered_noise.mean()
    return std*empowered_noise/empowered_noise.std()+mean

#%%
def power_noise_patches(p_row, p_col, f_row, f_col, power, num, mean=0, std=1):
    white_noise_field = np.random.normal(size=(f_row, f_col))
    power_filter = exponential_distance(f_row, f_col, (f_row-f_row%2)/2, (f_col-f_col%2)/2, power)
    xx, yy = np.meshgrid(range(f_row), range(f_col), sparse=True)
    xx = xx - (f_row-f_row%2)/2
    yy = yy - (f_col-f_col%2)/2
    power_filter[(xx<-p_row/2) | (xx>p_row/2) | (yy<-p_col/2) | (yy>p_col/2)] = 0 # remove high frequencies
    empowered_noise_field = filter_fft(white_noise_field, power_filter)
    empowered_noise_field -= empowered_noise_field.mean()
    empowered_noise_field = std*empowered_noise_field/empowered_noise_field.std() + mean
    
    empowered_noise_patches = np.zeros((p_row, p_col, num))
    template_patch = np.zeros((p_row, p_col))
    for i in range(num):
        empowered_noise_patches[:,:,i] = cut_randomly(empowered_noise_field, template_patch)
    
    return empowered_noise_patches, empowered_noise_field

#%%
def sine_wave(row, col, frequencies):
    # frequencies: cycles/image, (vertical, horizontal)
    ii, jj = np.meshgrid(range(row), range(col), sparse=True, indexing='ij')
    return -np.sin(2*np.pi/row*(ii-row/2)*frequencies[0]+2*np.pi/col*(jj-col/2)*frequencies[1])

#%%
def cosine_wave(row, col, frequencies):
    # frequencies: cycles/image, (vertical, horizontal)
    ii, jj = np.meshgrid(range(row), range(col), sparse=True, indexing='ij')
    return np.cos(2*np.pi/row*(ii-row/2)*frequencies[0]+2*np.pi/col*(jj-col/2)*frequencies[1])

#%%
def cosine_wave(row, col, frequencies):
    # frequencies: cycles/image, (vertical, horizontal)
    ii, jj = np.meshgrid(range(row), range(col), sparse=True, indexing='ij')
    return np.cos(2*np.pi/row*(ii-row/2)*frequencies[0]+2*np.pi/col*(jj-col/2)*frequencies[1])

#%%
def sawtooth_wave(row, col, frequencies, left_width=1):
    ii, jj = np.meshgrid(range(row), range(col), sparse=True, indexing='ij')
    return sawtooth(2*np.pi/row*(ii-row/2)*frequencies[0]+2*np.pi/col*(jj-col/2)*frequencies[1], left_width)

#%%
def sine_triangle_wave(row, col, frequencies):
    ii, jj = np.meshgrid(range(row), range(col), sparse=True, indexing='ij')
    return 2*np.abs(sawtooth(2*np.pi/row*(ii-row/2)*frequencies[0]+2*np.pi/col*(jj-col/2)*frequencies[1]+np.pi/2))-1

#%%
def cosine_triangle_wave(row, col, frequencies):
    ii, jj = np.meshgrid(range(row), range(col), sparse=True, indexing='ij')
    return 2*np.abs(sawtooth(2*np.pi/row*(ii-row/2)*frequencies[0]+2*np.pi/col*(jj-col/2)*frequencies[1]))-1

#%%
def sine_square_wave(row, col, frequencies, duty=0.5):
    ii, jj = np.meshgrid(range(row), range(col), sparse=True, indexing='ij')
    return -square(2*np.pi/row*(ii-row/2)*frequencies[0]+2*np.pi/col*(jj-col/2)*frequencies[1], duty)

#%%
def cosine_square_wave(row, col, frequencies, duty=0.5):
    ii, jj = np.meshgrid(range(row), range(col), sparse=True, indexing='ij')
    return square(2*np.pi/row*(ii-row/2)*frequencies[0]+2*np.pi/col*(jj-col/2)*frequencies[1]+np.pi/2, duty)

#%%
def gaussian_window(row, col, std):
    distance = distance_to_point(row, col, (row-row%2)/2, (col-col%2)/2)
    return np.exp(-distance**2/(2*np.pi*std))

#%%
def circ_mask_gaussian_window(row, col, cut_rad, std):
    distance = distance_to_point(row, col, (row-row%2)/2, (col-col%2)/2)
    return np.exp(-distance**2/(2*np.pi*std)) * circ_mask(row, col, cut_rad)

#%%
def drop_stretch_gaussian_window(row, col, cut_rad, std):
    window = circ_mask_gaussian_window(row, col, cut_rad, std)
    window[window>0] -= window[window>0].min()
    window /= window.max()
    return window

#%%
def flat_top_gaussian_window(row, col, cut_rad, std):
    distance = distance_to_point(row, col, (row-row%2)/2, (col-col%2)/2)
    window = np.exp(-distance**2/(2*np.pi*std)) * (1-circ_mask(row, col, cut_rad))
    window[window==0] = 1
    return window

#%%
def flat_top_circ_mask_gaussian_window(row, col, cut_rads, std):
    return flat_top_gaussian_window(row, col, cut_rads[0], std) * circ_mask(row, col, cut_rads[1])

#%%
def flat_top_drop_stretch_gaussian_window(row, col, cut_rads, std):
    window = flat_top_circ_mask_gaussian_window(row, col, cut_rads, std)
    window[window>0] -= window[window>0].min()
    window /= window.max()
    return window

#%%
def hann_window(row, col, cut_rad):
    distance = distance_to_point(row, col, (row-row%2)/2, (col-col%2)/2)
    return np.cos(np.pi*distance/(2*cut_rad))**2 * circ_mask(row, col, cut_rad)

#%%
def flat_top_hann_window(row, col, cut_rads):
    window = hann_window(row, col, cut_rads[1])
    window[circ_mask(row,col, cut_rads[0])==1] = 1
    return window
