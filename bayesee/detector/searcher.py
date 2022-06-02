#%%
import numpy as np
from numpy import random
from matplotlib.pyplot import *

from numba import cuda
if cuda.is_available():
    import cupy as cp
    import cupyx.scipy.signal as sg

#%%
from operation.nb2d import *
from imaging.image import *
from imaging.filter import *
from detector.observer import Observer

#%%
class Searcher(Observer):
    def __init__(self, method='TM', resp_func=None, whiten=None, weight=None, csf=None, uncer=None, inner_prod=None):
        
        super().__init__(method, resp_func, whiten, weight, csf, uncer, inner_prod)
       
class UncertainSearcher(Observer):
    def __init__(self, method='TM', resp_func=None, whiten=None, weight=None, csf=None, uncer=None, inner_prod=None):
        
        super().__init__(method, resp_func, whiten, weight, csf, uncer, inner_prod)
# tar_size = 96
# img_size = 516

# lambda img, tar: 
# TM.give_response(x, y)

# #%%
# base_path = Path(__file__).parent.parent
# os.chdir(base_path)

# from operation.numba2d import *

# #%%

# def sim_exponent(d, cc_t2):
#     X = random.normal(np.zeros_like(d), 1)
#     sim = d*X-0.5*(d**2)+np.abs(cc_t2)*(d**2)

#     return sim

# #%%
# def ELM_fixation_gpu(p_map, dp2):
#     p_map = cp.asarray(p_map)
#     dp2 = cp.asarray(dp2)
#     d2_p_map = sg.correlate2d(p_map, dp2, mode='same')
    
#     return cp.asnumpy(d2_p_map)

# #%%
# def ELM(target, t_mask, stimulus, s_mask, detect_map, loc_target=None, fix_prior=None, loc_prior=None, loc_prior0=None, gamma=0.5, max_fix=20):
#     # entropy limit minimiazation searcher
#     # stimulus consists of background (+ target)
#     # detect_map needs to be sufficiently large (safest: 2x stimulus.shape)

#     trow, tcol = target.shape
#     srow, scol = stimulus.shape
#     drow, dcol = detect_map.shape

#     # template (target shape)
#     template = target / nb2d_norm(target)
#     # simulated responses
#     R = np.zeros((srow, scol))
#     # detectability map overlapped with the stimulus
#     d = np.zeros((srow, scol))
#     # weight of prior map
#     w = np.zeros((srow, scol))
#     w0 = 1
#     # temporal integration of the exponent of w
#     ln_w = np.zeros((srow, scol))

#     # probability map of the current fixation
#     p_map  = np.zeros((max_fix+1, srow, scol))
#     p_map0 = np.zeros((max_fix+1))
#     # probability of the current fixation
#     p_fix = np.zeros(max_fix+1)
#     # the current fixation
#     loc_fix = np.zeros((max_fix+1, 2))

#     if fix_prior is None:
#         fix_prior = np.array((srow//2, scol//2))
    
#     if loc_prior is None:
#         if loc_prior0 is None:
#             loc_prior0 = 1/(nb2d_sum(t_mask)+1)
#             loc_prior = loc_prior0*np.ones((srow, scol))*t_mask
#         else:
#             loc_prior = (1-loc_prior0)*np.ones((srow, scol))*t_mask/nb2d_sum(t_mask)
#     elif loc_prior0 is None:
#             loc_prior0 = 0

#     p_map[0,:,:] = loc_prior
#     p_map0[0] = loc_prior0
#     p_fix[0] = loc_prior[fix_prior[0],fix_prior[1]]
#     loc_fix[0,:] = fix_prior
#     n_fix = 0
#     fix = loc_fix[0,:]

#     # precompute target cross-correlation fall-off
#     if loc_target is not None:
#         padded_template = np.zeros((srow, scol))
#         padded_template[int(loc_target[0]-trow//2):int(loc_target[0]+trow//2), int(loc_target[1]-tcol//2):int(loc_target[1]+tcol//2)] = template
#         cc_t2 = nb2d_cross_correlation(padded_template, template)
#     else:
#         cc_t2 = np.zeros((srow, scol))
    
#     while n_fix < max_fix and p_fix[n_fix] < gamma:
#         d = detect_map[int(drow/2-fix[0]):int(drow/2+(srow-fix[0])), int(dcol/2-fix[1]):int(dcol/2+(scol-fix[1]))]
#         ln_w = sim_exponent(d, cc_t2)
#         w = np.exp(ln_w)*s_mask

#         n_fix += 1

#         const_norm = nb2d_sum(p_map[n_fix-1,:,:]*w)+p_map0[n_fix-1]*w0
#         p_map_updated = (p_map[n_fix-1,:,:]*w) / const_norm
#         p_map[n_fix,:,:] = p_map_updated
#         p_map0[n_fix] = (p_map0[n_fix-1]*w0) / const_norm

#         if p_map_updated.max()>p_map0[n_fix]:
#             p_fix[n_fix] = p_map_updated.max()
            
#         else:
#             p_fix[n_fix] = p_map0[n_fix]

#         d2_p_map = nb2d_cross_correlation(p_map_updated, detect_map**2)
#         loc_candidates = np.where(d2_p_map==d2_p_map.max())
        
#         rand_id = random.randint(len(loc_candidates[0]))
#         loc_fix[n_fix,:] = [loc_candidates[0][rand_id], loc_candidates[1][rand_id]]
#         fix = loc_fix[n_fix,:]

#     return p_fix, loc_fix, p_map, p_map0

# #%%
# def ELM_gpu(target, t_mask, stimulus, s_mask, detect_map, loc_target=None, fix_prior=None, loc_prior=None, loc_prior0=None, gamma=0.5, max_fix=20):
#     # entropy limit minimization searcher
#     # stimulus consists of background (+ target)
#     # detect_map needs to be sufficiently large (safest: 2x stimulus.shape)

#     trow, tcol = target.shape
#     srow, scol = stimulus.shape
#     drow, dcol = detect_map.shape

#     # template (target shape)
#     template = cp.asarray(target / nb2d_norm(target))
#     # simulated responses
#     R = np.zeros((srow, scol))
#     # detectability map overlapped with the stimulus
#     d = np.zeros((srow, scol))
#     # weight of prior map
#     w = np.zeros((srow, scol))
#     w0 = 1
#     # temporal integration of the exponent of w
#     ln_w = np.zeros((srow, scol))

#     # probability map of the current fixation
#     p_map  = np.zeros((max_fix+1, srow, scol))
#     p_map0 = np.zeros((max_fix+1))
#     # probability of the current fixation
#     p_fix = np.zeros(max_fix+1)
#     # the current fixation
#     loc_fix = np.zeros((max_fix+1, 2))

#     if fix_prior is None:
#         fix_prior = np.array((srow//2, scol//2))
    
#     if loc_prior is None:
#         if loc_prior0 is None:
#             loc_prior0 = 1/(nb2d_sum(t_mask)+1)
#             loc_prior = loc_prior0*np.ones((srow, scol))*t_mask
#         else:
#             loc_prior = (1-loc_prior0)*np.ones((srow, scol))*t_mask/nb2d_sum(t_mask)
#     elif loc_prior0 is None:
#             loc_prior0 = 0

#     p_map[0,:,:] = loc_prior
#     p_map0[0] = loc_prior0
#     p_fix[0] = loc_prior[fix_prior[0],fix_prior[1]]
#     loc_fix[0,:] = fix_prior
#     fix = loc_fix[0,:]
#     n_fix = 0

#     #precompute target cross-correlation fall-off
#     if loc_target is not None:
#         padded_template = cp.zeros((srow, scol))
#         padded_template[int(loc_target[0]-trow//2):int(loc_target[0]+trow//2), int(loc_target[1]-tcol//2):int(loc_target[1]+tcol//2)] = template
#         cc_t2 = cp.asnumpy(sg.correlate2d(padded_template, template, mode='same'))
#     else:
#         cc_t2 = np.zeros((srow, scol))
    
#     while n_fix < max_fix and p_fix[n_fix] < gamma:
#         d = detect_map[int(drow/2-fix[0]):int(drow/2+(srow-fix[0])), int(dcol/2-fix[1]):int(dcol/2+(scol-fix[1]))]
#         ln_w = sim_exponent(d, cc_t2)
#         w = np.exp(ln_w)*s_mask

#         n_fix += 1

#         const_norm = nb2d_sum(p_map[n_fix-1,:,:]*w)+p_map0[n_fix-1]*w0
#         p_map_updated = (p_map[n_fix-1,:,:]*w) / const_norm
#         p_map[n_fix,:,:] = p_map_updated
#         p_map0[n_fix] = (p_map0[n_fix-1]*w0) / const_norm

#         if p_map_updated.max()>p_map0[n_fix]:
#             p_fix[n_fix] = p_map_updated.max()
            
#         else:
#             p_fix[n_fix] = p_map0[n_fix]

#         d2_p_map = ELM_fixation_gpu(p_map_updated, detect_map**2)
#         loc_candidates = np.where(d2_p_map==d2_p_map.max())
        
#         rand_id = random.randint(len(loc_candidates[0]))
#         loc_fix[n_fix,:] = [loc_candidates[0][rand_id], loc_candidates[1][rand_id]]
#         fix = loc_fix[n_fix,:]

#     return p_fix, loc_fix, p_map, p_map0

# #%%
# def MAP(target, t_mask, stimulus, s_mask, detect_map, loc_target=None, fix_prior=None, loc_prior=None, loc_prior0=None, gamma=0.5, max_fix=20):
#     # Maximum a posteriori searcher
#     # stimulus consists of background (+ target)
#     # detect_map needs to be sufficiently large (safest: 2x stimulus.shape)

#     trow, tcol = target.shape
#     srow, scol = stimulus.shape
#     drow, dcol = detect_map.shape

#     # template (target shape)
#     template = target / nb2d_norm(target)
#     # simulated responses
#     R = np.zeros((srow, scol))
#     # detectability map overlapped with the stimulus
#     d = np.zeros((srow, scol))
#     # weight of prior map
#     w = np.zeros((srow, scol))
#     w0 = 1
#     # temporal integration of the exponent of w
#     ln_w = np.zeros((srow, scol))

#     # probability map of the current fixation
#     p_map  = np.zeros((max_fix+1, srow, scol))
#     p_map0 = np.zeros((max_fix+1))
#     # probability of the current fixation
#     p_fix = np.zeros(max_fix+1)
#     # the current fixation
#     loc_fix = np.zeros((max_fix+1, 2))

#     if fix_prior is None:
#         fix_prior = np.array((srow//2, scol//2))
    
#     if loc_prior is None:
#         if loc_prior0 is None:
#             loc_prior0 = 1/(nb2d_sum(t_mask)+1)
#             loc_prior = loc_prior0*np.ones((srow, scol))*t_mask
#         else:
#             loc_prior = (1-loc_prior0)*np.ones((srow, scol))*t_mask/nb2d_sum(t_mask)
#     elif loc_prior0 is None:
#             loc_prior0 = 0

#     p_map[0,:,:] = loc_prior
#     p_map0[0] = loc_prior0
#     p_fix[0] = loc_prior[fix_prior[0],fix_prior[1]]
#     loc_fix[0,:] = fix_prior
#     n_fix = 0
#     fix = loc_fix[0,:]

#     # precompute target cross-correlation fall-off
#     if loc_target is not None:
#         padded_template = np.zeros((srow, scol))
#         padded_template[int(loc_target[0]-trow//2):int(loc_target[0]+trow//2), int(loc_target[1]-tcol//2):int(loc_target[1]+tcol//2)] = template
#         cc_t2 = nb2d_cross_correlation(padded_template, template)
#     else:
#         cc_t2 = np.zeros((srow, scol))
    
#     while n_fix < max_fix and p_fix[n_fix] < gamma:
#         d = detect_map[int(drow/2-fix[0]):int(drow/2+(srow-fix[0])), int(dcol/2-fix[1]):int(dcol/2+(scol-fix[1]))]
#         ln_w = sim_exponent(d, cc_t2)
#         w = np.exp(ln_w)*s_mask

#         n_fix += 1

#         const_norm = nb2d_sum(p_map[n_fix-1,:,:]*w)+p_map0[n_fix-1]*w0
#         p_map_updated = (p_map[n_fix-1,:,:]*w) / const_norm
#         p_map[n_fix,:,:] = p_map_updated
#         p_map0[n_fix] = (p_map0[n_fix-1]*w0) / const_norm

#         if p_map_updated.max()>p_map0[n_fix]:
#             p_fix[n_fix] = p_map_updated.max()
            
#         else:
#             p_fix[n_fix] = p_map0[n_fix]


#         loc_candidates = np.where(p_map_updated==p_map_updated.max())
#         rand_id = random.randint(len(loc_candidates[0]))
#         loc_fix[n_fix,:] = [loc_candidates[0][rand_id], loc_candidates[1][rand_id]]
#         fix = loc_fix[n_fix,:]

#     return p_fix, loc_fix, p_map, p_map0

# #%%
# def MAP_gpu(target, t_mask, stimulus, s_mask, detect_map, loc_target=None, fix_prior=None, loc_prior=None, loc_prior0=None, gamma=0.5, max_fix=20):
#     # Maximum a posteriori searcher
#     # stimulus consists of background (+ target)
#     # detect_map needs to be sufficiently large (safest: 2x stimulus.shape)

#     trow, tcol = target.shape
#     srow, scol = stimulus.shape
#     drow, dcol = detect_map.shape

#     # template (target shape)
#     template = cp.asarray(target / nb2d_norm(target))
#     # simulated responses
#     R = np.zeros((srow, scol))
#     # detectability map overlapped with the stimulus
#     d = np.zeros((srow, scol))
#     # weight of prior map
#     w = np.zeros((srow, scol))
#     w0 = 1
#     # temporal integration of the exponent of w
#     ln_w = np.zeros((srow, scol))

#     # probability map of the current fixation
#     p_map  = np.zeros((max_fix+1, srow, scol))
#     p_map0 = np.zeros((max_fix+1))
#     # probability of the current fixation
#     p_fix = np.zeros(max_fix+1)
#     # the current fixation
#     loc_fix = np.zeros((max_fix+1, 2))

#     if fix_prior is None:
#         fix_prior = np.array((srow//2, scol//2))
    
#     if loc_prior is None:
#         if loc_prior0 is None:
#             loc_prior0 = 1/(nb2d_sum(t_mask)+1)
#             loc_prior = loc_prior0*np.ones((srow, scol))*t_mask
#         else:
#             loc_prior = (1-loc_prior0)*np.ones((srow, scol))*t_mask/nb2d_sum(t_mask)
#     elif loc_prior0 is None:
#             loc_prior0 = 0

#     p_map[0,:,:] = loc_prior
#     p_map0[0] = loc_prior0
#     p_fix[0] = loc_prior[fix_prior[0],fix_prior[1]]
#     loc_fix[0,:] = fix_prior
#     fix = loc_fix[0,:]
#     n_fix = 0

#     #precompute target cross-correlation fall-off
#     if loc_target is not None:
#         padded_template = cp.zeros((srow, scol))
#         padded_template[int(loc_target[0]-trow//2):int(loc_target[0]+trow//2), int(loc_target[1]-tcol//2):int(loc_target[1]+tcol//2)] = template
#         cc_t2 = cp.asnumpy(sg.correlate2d(padded_template, template, mode='same'))
#     else:
#         cc_t2 = np.zeros((srow, scol))
    
#     while n_fix < max_fix and p_fix[n_fix] < gamma:
#         d = detect_map[int(drow/2-fix[0]):int(drow/2+(srow-fix[0])), int(dcol/2-fix[1]):int(dcol/2+(scol-fix[1]))]
#         ln_w = sim_exponent(d, cc_t2)
#         w = np.exp(ln_w)*s_mask

#         n_fix += 1

#         const_norm = nb2d_sum(p_map[n_fix-1,:,:]*w)+p_map0[n_fix-1]*w0
#         p_map_updated = (p_map[n_fix-1,:,:]*w) / const_norm
#         p_map[n_fix,:,:] = p_map_updated
#         p_map0[n_fix] = (p_map0[n_fix-1]*w0) / const_norm

#         if p_map_updated.max()>p_map0[n_fix]:
#             p_fix[n_fix] = p_map_updated.max()
            
#         else:
#             p_fix[n_fix] = p_map0[n_fix]


#         loc_candidates = np.where(p_map_updated==p_map_updated.max())
#         rand_id = random.randint(len(loc_candidates[0]))
#         loc_fix[n_fix,:] = [loc_candidates[0][rand_id], loc_candidates[1][rand_id]]
#         fix = loc_fix[n_fix,:]

#     return p_fix, loc_fix, p_map, p_map0

# #%%
# def MAP2(target, t_mask, stimulus, s_mask, detect_map, loc_target=None, fix_prior=None, loc_prior=None, loc_prior0=None, gamma=0.5, max_fix=20):
#     # Maximum a posteriori searcher
#     # stimulus consists of background (+ target)
#     # detect_map needs to be sufficiently large (safest: 2x stimulus.shape)

#     trow, tcol = target.shape
#     srow, scol = stimulus.shape
#     drow, dcol = detect_map.shape

#     # template (target shape)
#     template = target / nb2d_norm(target)
#     # simulated responses
#     R = np.zeros((srow, scol))
#     # detectability map overlapped with the stimulus
#     d = np.zeros((srow, scol))
#     # weight of prior map
#     w = np.zeros((srow, scol))
#     w0 = 1
#     # temporal integration of the exponent of w
#     ln_w = np.zeros((srow, scol))

#     # probability map of the current fixation
#     p_map  = np.zeros((max_fix+1, srow, scol))
#     p_map0 = np.zeros((max_fix+1))
#     # probability of the current fixation
#     p_fix = np.zeros(max_fix+1)
#     # the current fixation
#     loc_fix = np.zeros((max_fix+1, 2))

#     if fix_prior is None:
#         fix_prior = np.array((srow//2, scol//2))
    
#     if loc_prior is None:
#         if loc_prior0 is None:
#             loc_prior0 = 1/(nb2d_sum(t_mask)+1)
#             loc_prior = loc_prior0*np.ones((srow, scol))*t_mask
#         else:
#             loc_prior = (1-loc_prior0)*np.ones((srow, scol))*t_mask/nb2d_sum(t_mask)
#     elif loc_prior0 is None:
#             loc_prior0 = 0

#     p_map[0,:,:] = loc_prior
#     p_map0[0] = loc_prior0
#     p_fix[0] = loc_prior[fix_prior[0],fix_prior[1]]
#     loc_fix[0,:] = fix_prior
#     n_fix = 0
#     fix = loc_fix[0,:]

#     # precompute target cross-correlation fall-off
#     if loc_target is not None:
#         padded_template = np.zeros((srow, scol))
#         padded_template[int(loc_target[0]-trow//2):int(loc_target[0]+trow//2), int(loc_target[1]-tcol//2):int(loc_target[1]+tcol//2)] = template
#         cc_t2 = nb2d_cross_correlation(padded_template, template)
#     else:
#         cc_t2 = np.zeros((srow, scol))
    
#     while n_fix < max_fix and p_fix[n_fix] < gamma:
#         d = detect_map[int(drow/2-fix[0]):int(drow/2+(srow-fix[0])), int(dcol/2-fix[1]):int(dcol/2+(scol-fix[1]))]
#         ln_w = sim_exponent(d, cc_t2)
#         w = np.exp(ln_w)*s_mask

#         n_fix += 1

#         const_norm = nb2d_sum(p_map[n_fix-1,:,:]*w)+p_map0[n_fix-1]*w0
#         p_map_updated = (p_map[n_fix-1,:,:]*w) / const_norm
#         p_map[n_fix,:,:] = p_map_updated
#         p_map0[n_fix] = (p_map0[n_fix-1]*w0) / const_norm

#         if p_map_updated.max()>p_map0[n_fix]:
#             p_fix[n_fix] = p_map_updated.max()
#             loc_candidates = np.where(p_map_updated==p_map_updated.max())
            
#         else:
#             p_fix[n_fix] = p_map0[n_fix]
#             loc_candidates = np.where(t_mask==1)
        
#         rand_id = random.randint(len(loc_candidates[0]))
#         loc_fix[n_fix,:] = [loc_candidates[0][rand_id], loc_candidates[1][rand_id]]
#         fix = loc_fix[n_fix,:]

#     return p_fix, loc_fix, p_map, p_map0

# #%%
# def MAP2_gpu(target, t_mask, stimulus, s_mask, detect_map, loc_target=None, fix_prior=None, loc_prior=None, loc_prior0=None, gamma=0.5, max_fix=20):
#     # Maximum a posteriori searcher
#     # stimulus consists of background (+ target)
#     # detect_map needs to be sufficiently large (safest: 2x stimulus.shape)

#     trow, tcol = target.shape
#     srow, scol = stimulus.shape
#     drow, dcol = detect_map.shape

#     # template (target shape)
#     template = cp.asarray(target / nb2d_norm(target))
#     # simulated responses
#     R = np.zeros((srow, scol))
#     # detectability map overlapped with the stimulus
#     d = np.zeros((srow, scol))
#     # weight of prior map
#     w = np.zeros((srow, scol))
#     w0 = 1
#     # temporal integration of the exponent of w
#     ln_w = np.zeros((srow, scol))

#     # probability map of the current fixation
#     p_map  = np.zeros((max_fix+1, srow, scol))
#     p_map0 = np.zeros((max_fix+1))
#     # probability of the current fixation
#     p_fix = np.zeros(max_fix+1)
#     # the current fixation
#     loc_fix = np.zeros((max_fix+1, 2))

#     if fix_prior is None:
#         fix_prior = np.array((srow//2, scol//2))
    
#     if loc_prior is None:
#         if loc_prior0 is None:
#             loc_prior0 = 1/(nb2d_sum(t_mask)+1)
#             loc_prior = loc_prior0*np.ones((srow, scol))*t_mask
#         else:
#             loc_prior = (1-loc_prior0)*np.ones((srow, scol))*t_mask/nb2d_sum(t_mask)
#     elif loc_prior0 is None:
#             loc_prior0 = 0

#     p_map[0,:,:] = loc_prior
#     p_map0[0] = loc_prior0
#     p_fix[0] = loc_prior[fix_prior[0],fix_prior[1]]
#     loc_fix[0,:] = fix_prior
#     fix = loc_fix[0,:]
#     n_fix = 0

#     #precompute target cross-correlation fall-off
#     if loc_target is not None:
#         padded_template = cp.zeros((srow, scol))
#         padded_template[int(loc_target[0]-trow//2):int(loc_target[0]+trow//2), int(loc_target[1]-tcol//2):int(loc_target[1]+tcol//2)] = template
#         cc_t2 = cp.asnumpy(sg.correlate2d(padded_template, template, mode='same'))
#     else:
#         cc_t2 = np.zeros((srow, scol))
    
#     while n_fix < max_fix and p_fix[n_fix] < gamma:
#         d = detect_map[int(drow/2-fix[0]):int(drow/2+(srow-fix[0])), int(dcol/2-fix[1]):int(dcol/2+(scol-fix[1]))]
#         ln_w = sim_exponent(d, cc_t2)
#         w = np.exp(ln_w)*s_mask

#         n_fix += 1

#         const_norm = nb2d_sum(p_map[n_fix-1,:,:]*w)+p_map0[n_fix-1]*w0
#         p_map_updated = (p_map[n_fix-1,:,:]*w) / const_norm
#         p_map[n_fix,:,:] = p_map_updated
#         p_map0[n_fix] = (p_map0[n_fix-1]*w0) / const_norm

#         if p_map_updated.max()>p_map0[n_fix]:
#             p_fix[n_fix] = p_map_updated.max()
#             loc_candidates = np.where(p_map_updated==p_map_updated.max())
            
#         else:
#             p_fix[n_fix] = p_map0[n_fix]
#             loc_candidates = np.where(t_mask==1)
        
#         rand_id = random.randint(len(loc_candidates[0]))
#         loc_fix[n_fix,:] = [loc_candidates[0][rand_id], loc_candidates[1][rand_id]]
#         fix = loc_fix[n_fix,:]

#     return p_fix, loc_fix, p_map, p_map0
