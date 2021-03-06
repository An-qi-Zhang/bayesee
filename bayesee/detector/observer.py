#%%
import numpy as np
from matplotlib.pyplot import *

from numba import cuda
if cuda.is_available():
    import cupy as cp
    import cupyx.scipy.signal as sg

#%%
from bayesee.operation.nb2d import *
from bayesee.imaging.image import *
from bayesee.imaging.filter import *
    
#%%
class Observer:
    def __init__(self, method='TM', resp_func=None, whiten=None, weight=None, csf=None, uncer=None, inner_prod=None):
        self.method = method
        self.resp_func = resp_func
        self.whiten = whiten
        self.weight = weight
        self.csf = csf
        self.uncer = uncer
        self.inner_prod = inner_prod
        
        if inner_prod is None:
            def x_size_conv(x, y):
                if cuda.is_available():
                    return sg.correlate2d(x, y, mode='same')
                else:
                    return nb2cross(x,y)
                
            self.inner_prod = x_size_conv
        else:
            self.inner_prod = inner_prod
        
        if self.method == 'TM':
            self.respond = lambda img, tar: (self.inner_prod(img, tar), img, tar)
        elif self.method == 'WTM':
            def func(img, tar, whiten):
                if whiten is None:
                    whiten_img = 1/np.absolute(np.fft.fftshift(np.fft.fft2(img)))
                    whiten_img[whiten_img.shape[0]//2, whiten_img.shape[1]//2] = 0
                elif type(whiten) == int or type(whiten) == float:
                    whiten_img = exponential_distance(*img.shape, img.shape[0]//2, img.shape[1]//2, whiten) # whitening exponent is the additive inverse of the image exponent
                elif whiten.shape == img.shape:
                    whiten_img = whiten
                else:
                    raise ValueError('Unable to interpret the whitening parameter.')
                    
                whiten_tar = cut_center(whiten_img, tar)
                
                whitened_img = filter_fft(img, whiten_img)
                whitened_tar = filter_fft(tar, whiten_tar)
                
                return self.inner_prod(whitened_img, whitened_tar), whitened_img, whitened_tar

            self.respond = lambda img, tar: func(img, tar, whiten)
        elif self.method == 'RTM':
            def func(img, tar, weight):
                if weight.shape == img.shape:
                    weight_img = weight # Weighting matrix is the multiplicative inverse of the local standard deviation matrix of the image.
                elif type(weight) == int or type(weight) == float: 
                    weight_img = 1/local_std(img, weight)
                else:
                    raise ValueError('Unable to interpret the weighting parameter.')
                    
                weight_tar = cut_center(weight_img, tar)
                
                weighted_img = (img-img.mean())*weight_img
                weighted_tar = (tar-tar.mean())*weight_tar
                
                return self.inner_prod(weighted_img, weighted_tar), weighted_img, weighted_tar
                
            self.respond = lambda img, tar: func(img, tar, weight)
        elif self.method == 'RWTM':
            def func(img, tar, whiten, weight):
                if whiten is None:
                    whiten_img = 1/np.absolute(np.fft.fftshift(np.fft.fft2(img)))
                    whiten_img[whiten_img.shape[0]//2, whiten_img.shape[1]//2] = 0
                elif type(whiten) == int or type(whiten) == float:
                    whiten_img = exponential_distance(*img.shape, img.shape[0]//2, img.shape[1]//2, whiten) # whitening exponent is the additive inverse of the image exponent
                elif whiten.shape == img.shape:
                    whiten_img = whiten
                else:
                    raise ValueError('Unable to interpret the whitening parameter.')
                
                whiten_tar = cut_center(whiten_img, tar)
                
                if weight.shape == img.shape:
                    weight_img = weight # Weighting matrix is the multiplicative inverse of the local variance matrix of the image.
                elif type(weight) == int or type(weight) == float:
                    weight_img = 1/local_std(img, weight)
                else:
                    raise ValueError('Unable to interpret the weighting parameter.')
                    
                weight_tar = cut_center(weight_img, tar)
                
                whitened_img = filter_fft(img, whiten_img)
                whitened_tar = filter_fft(tar, whiten_tar)
                weighted_whitened_img = (whitened_img-whitened_img.mean())*weight_img
                weighted_whitened_tar = (whitened_tar-whitened_tar.mean())*weight_tar
                
                return self.inner_prod(weighted_whitened_img, weighted_whitened_tar), weighted_whitened_img, weighted_whitened_tar
                
            self.respond = lambda img, tar: func(img, tar, whiten, weight)
        elif self.method == 'WRTM':
            def func(img, tar, whiten, weight):
                if whiten is None:
                    whiten_img = 1/np.absolute(np.fft.fftshift(np.fft.fft2(img)))
                    whiten_img[whiten_img.shape[0]//2, whiten_img.shape[1]//2] = 0
                elif type(whiten) == int or type(whiten) == float:
                    whiten_img = exponential_distance(*img.shape, img.shape[0]//2, img.shape[1]//2, whiten) # whitening exponent is the additive inverse of the image exponent
                elif whiten.shape == img.shape:
                    whiten_img = whiten
                else:
                    raise ValueError('Unable to interpret the whitening parameter.')
                
                whiten_tar = cut_center(whiten_img, tar)
                
                if weight.shape == img.shape:
                    weight_img = weight # Weighting matrix is the multiplicative inverse of the local variance matrix of the image.
                elif type(weight) == int or type(weight) == float:
                    weight_img = 1/local_std(img, weight)
                else:
                    raise ValueError('Unable to interpret the weighting parameter.')
                    
                weight_tar = cut_center(weight_img, tar)
                
                whitened_weighted_img = filter_fft((img-img.mean())*weight_img, whiten_img) 
                whitened_weighted_tar = filter_fft((tar-tar.mean())*weight_tar, whiten_tar)
                
                return self.inner_prod(whitened_weighted_img, whitened_weighted_tar), whitened_weighted_img, whitened_weighted_tar
                
            self.respond = lambda img, tar: func(img, tar, whiten, weight)
        elif self.method == 'ETM':
            def func(img, tar, csf):
                csf_img_mat, csf_tar_mat = csf_filter(*img.shape,**csf), csf_filter(*tar.shape,**csf)
                csf_img_mat /= csf_img_mat.max()
                csf_tar_mat /= csf_tar_mat.max()
                csfed_img, csfed_tar = filter_fft(img, csf_img_mat), filter_fft(tar, csf_tar_mat)
                return self.inner_prod(csfed_img, csfed_tar), csfed_img, csfed_tar
            
            self.respond = lambda img, tar: func(img, tar, csf)
        elif self.method == 'RETM':
            def func(img, tar, weight, csf):
                csf_img_mat, csf_tar_mat = csf_filter(*img.shape,**csf), csf_filter(*tar.shape,**csf)
                csf_img_mat /= csf_img_mat.max()
                csf_tar_mat /= csf_tar_mat.max()
                csfed_img, csfed_tar = filter_fft(img, csf_img_mat), filter_fft(tar, csf_tar_mat)
                
                if weight.shape == img.shape:
                    weight_img = weight # Weighting matrix is the multiplicative inverse of the local standard deviation matrix of the image.
                elif type(weight) == int or type(weight) == float:
                    weight_img = 1/local_std(csfed_img, weight)
                else:
                    raise ValueError('Unable to interpret the weighting parameter.')
                
                weight_tar = cut_center(weight_img, tar)
                
                weighted_csfed_img = (csfed_img-csfed_img.mean())*weight_img
                weighted_csfed_tar = (csfed_tar-csfed_tar.mean())*weight_tar
                
                return self.inner_prod(weighted_csfed_img, weighted_csfed_tar), weighted_csfed_img, weighted_csfed_tar
            
            self.respond = lambda img, tar: func(img, tar, weight, csf)
        elif self.method == 'ERTM':
            def func(img, tar, weight, csf):
                if weight.shape == img.shape:
                    weight_img = weight # Weighting matrix is the multiplicative inverse of the local standard deviation matrix of the image.
                elif type(weight) == int or type(weight) == float:
                    weight_img = 1/local_std(img, weight)
                else:
                    raise ValueError('Unable to interpret the weighting parameter.')
                    
                weight_tar = cut_center(weight_img, tar)
                
                csf_img_mat, csf_tar_mat = csf_filter(*img.shape,**csf), csf_filter(*tar.shape,**csf)
                csf_img_mat /= csf_img_mat.max()
                csf_tar_mat /= csf_tar_mat.max()
                csfed_weighted_img, csfed_weighted_tar = filter_fft((img-img.mean())*weight_img, csf_img_mat), filter_fft((tar-tar.mean())*weight_tar, csf_tar_mat)
                
                return self.inner_prod(csfed_weighted_img, csfed_weighted_tar), csfed_weighted_img, csfed_weighted_tar
            
            self.respond = lambda img, tar: func(img, tar, weight, csf)
        elif self.method == 'DIY':
            self.respond = resp_func # return response, filtered image, and filtered target
            
    def give_response(self, img, tar):
        self.response, _, _ = self.respond(img, tar)
        return self.response
