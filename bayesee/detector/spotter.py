#%%
import numpy as np
from numpy import random
from pprint import pprint as pp
from matplotlib.pyplot import *
import scipy as sp

from numba import cuda
if cuda.is_available():
    import cupy as cp
    import cupyx.scipy.signal as sg

#%%
from detector.observer import Observer
from imaging.image import *
from imaging.filter import *
from operation.nb2d import *

#%%
class Spotter(Observer):
    def __init__(self, method='TM', resp_func=None, whiten=None, weight=None, csf=None, uncer=None, inner_prod=None):
        
        super().__init__(method, resp_func, whiten, weight, csf, uncer, inner_prod)
    
        if inner_prod is None:
            def x_size_dot(img, tar):
                img_cut = cut_center(img, tar)
                if cuda.is_available():
                    return cp.einsum('ij,ij', img_cut, tar)
                else:
                    return nb2dot(img_cut, tar)
                
            self.inner_prod = x_size_dot
        else:
            self.inner_prod = inner_prod
            
#%%
class UncertainSpotter(Observer):
    def __init__(self, method='UTM', resp_func=None, whiten=None, weight=None, csf=None, uncer=None, inner_prod=None):
        if method[0] != 'U':
            raise ValueError('Please include uncertainty explicitly.')
        
        if inner_prod is None:
            def x_size_dot(img, tar, upleft_i, upleft_j):
                img_cut = cut_by_coord(img, tar, upleft_i, upleft_j)
                if cuda.is_available():
                    return cp.einsum('ij,ij->', img_cut, tar)
                else:
                    return nb2dot(img_cut, tar)
                
            inner_prod = x_size_dot

        super().__init__(method, resp_func, whiten, weight, csf, uncer, inner_prod)
        
        def give_response(self, img, tar, uncer):
            self.responses, self.pp_img, self.pp_tar = self.respond(img, tar)
            if uncer['focus'] == "max":
                self.response = self.responses.max()
            elif uncer['focus'] == "sum":
                self.response = self.responses.sum()
            return self.response
        
        if self.method == 'UTM':
            def func(img, tar, uncer):
                size = uncer['size']
                dist = distance_to_point(*img.shape, (img.shape[0]-img.shape[0]%2)/2,(img.shape[1]-img.shape[1]%2)/2)
                circ = circ_mask(*img.shape, size)
                uncer_mat = uncer['func'](dist) * circ
                uncer_mat /= np.einsum('ij->', uncer_mat)
                
                if uncer['info'] == "sample":
                    responses = np.zeros_like(img)
                    rand_mat = np.random.rand(*img.shape)
                    for i in np.ndindex(img):
                        if uncer_mat[i] > rand_mat[i]:
                            responses[i]=self.inner_prod(img, tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                elif uncer['info'] == "prior":
                    likely = np.zeros_like(img)
                    for i in np.ndindex(img):
                        if circ[i] == 1:
                            likely[i] = self.inner_prod(img, tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                    responses = uncer_mat*likely
                    responses[responses>0] = np.log(responses[responses>0])
                
                return responses, img, tar
                        
            self.respond = lambda img, tar: func(img, tar, uncer)
        elif self.method == 'UWTM':
            def func(img, tar, whiten, uncer):
                if whiten is None:
                    whiten_img = 1/np.absolute(np.fft.fftshift(np.fft.fft2(img)))
                elif whiten.isnumeric():
                    whiten_img = exponential_distance(*img.shape, (img.shape[0]-img.shape[0]%2)/2.0, (img.shape[1]-img.shape[1]%2)/2.0, whiten) # whitening exponent is the additive inverse of the image exponent
                elif whiten.shape == img.shape:
                    whiten_img = whiten
                else:
                    raise ValueError('Unable to interpret the whitening parameter.')
                    
                whiten_tar = cut_center(whiten_img, tar)
                
                whitened_img = filter_fft(img, whiten_img)
                whitened_tar = filter_fft(tar, whiten_tar)

                size = uncer['size']
                dist = distance_to_point(*img.shape, (img.shape[0]-img.shape[0]%2)/2,(img.shape[1]-img.shape[1]%2)/2)
                circ = circ_mask(*img.shape, size)
                uncer_mat = uncer['func'](dist) * circ
                uncer_mat /= np.einsum('ij->', uncer_mat)
                
                if uncer['info'] == "sample":
                    responses = np.zeros_like(img)
                    rand_mat = np.random.rand(*img.shape)
                    for i in np.ndindex(img):
                        if uncer_mat[i] > rand_mat[i]:
                            responses[i]=self.inner_prod(whitened_img, whitened_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                elif uncer['info'] == "prior":
                    likely = np.zeros_like(img)
                    for i in np.ndindex(img):
                        if circ[i] == 1:
                            likely[i] = self.inner_prod(whitened_img, whitened_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                    responses = uncer_mat*likely
                    responses[responses>0] = np.log(responses[responses>0])
                
                return responses, whitened_img, whitened_tar
                        
            self.respond = lambda img, tar: func(img, tar, whiten, uncer)
        elif self.method == 'URTM':
            def func(img, tar, weight, uncer):
                if weight.shape == img.shape:
                    weight_img = weight # Weighting matrix is the multiplicative inverse of the local standard deviation matrix of the image.
                elif weight.isnumeric():
                    weight_img = 1/local_std(img, weight)
                else:
                    raise ValueError('Unable to interpret the weighting parameter.')
                    
                weight_tar = cut_center(weight_img, tar)
                
                weighted_img = img*weight_img
                weighted_tar = tar*weight_tar
                
                size = uncer['size']
                dist = distance_to_point(*img.shape, (img.shape[0]-img.shape[0]%2)/2,(img.shape[1]-img.shape[1]%2)/2)
                circ = circ_mask(*img.shape, size)
                uncer_mat = uncer['func'](dist) * circ
                uncer_mat /= np.einsum('ij->', uncer_mat)
                
                if uncer['info'] == "sample":
                    responses = np.zeros_like(img)
                    rand_mat = np.random.rand(*img.shape)
                    for i in np.ndindex(img):
                        if uncer_mat[i] > rand_mat[i]:
                            responses[i]=self.inner_prod(weighted_img, weighted_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                elif uncer['info'] == "prior":
                    likely = np.zeros_like(img)
                    for i in np.ndindex(img):
                        if circ[i] == 1:
                            likely[i] = self.inner_prod(weighted_img, weighted_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                    responses = uncer_mat*likely
                    responses[responses>0] = np.log(responses[responses>0])
                
                return responses, weighted_img, weighted_tar
                                        
            self.respond = lambda img, tar: func(img, tar, weight, uncer)
        elif self.method == 'URWTM':
            def func(img, tar, whiten, weight, uncer):
                if whiten is None:
                    whiten_img = 1/np.absolute(np.fft.fftshift(np.fft.fft2(img)))
                elif whiten.isnumeric():
                    whiten_img = exponential_distance(*img.shape, (img.shape[0]-img.shape[0]%2)/2.0, (img.shape[1]-img.shape[1]%2)/2.0, whiten) # whitening exponent is the additive inverse of the image exponent
                elif whiten.shape == img.shape:
                    whiten_img = whiten
                else:
                    raise ValueError('Unable to interpret the whitening parameter.')
                
                whiten_tar = cut_center(whiten_img, tar)
                
                if weight.shape == img.shape:
                    weight_img = weight # Weighting matrix is the multiplicative inverse of the local variance matrix of the image.
                elif weight.isnumeric():
                    weight_img = 1/local_std(img, weight)
                else:
                    raise ValueError('Unable to interpret the weighting parameter.')
                    
                weight_tar = cut_center(weight_img, tar)
                
                weighted_whitened_img = filter_fft(img, whiten_img) * weight_img
                weighted_whitened_tar = filter_fft(tar, whiten_tar) * weight_tar
                
                size = uncer['size']
                dist = distance_to_point(*img.shape, (img.shape[0]-img.shape[0]%2)/2,(img.shape[1]-img.shape[1]%2)/2)
                circ = circ_mask(*img.shape, size)
                uncer_mat = uncer['func'](dist) * circ
                uncer_mat /= np.einsum('ij->', uncer_mat)
                
                if uncer['info'] == "sample":
                    responses = np.zeros_like(img)
                    rand_mat = np.random.rand(*img.shape)
                    for i in np.ndindex(img):
                        if uncer_mat[i] > rand_mat[i]:
                            responses[i]=self.inner_prod(weighted_whitened_img, weighted_whitened_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                elif uncer['info'] == "prior":
                    likely = np.zeros_like(img)
                    for i in np.ndindex(img):
                        if circ[i] == 1:
                            likely[i] = self.inner_prod(weighted_whitened_img, weighted_whitened_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                    responses = uncer_mat*likely
                    responses[responses>0] = np.log(responses[responses>0])
                
                return responses, weighted_whitened_img, weighted_whitened_tar
                
            self.respond = lambda img, tar: func(img, tar, whiten, weight, uncer)
        elif self.method == 'UWRTM':
            def func(img, tar, whiten, weight, uncer):
                if whiten is None:
                    whiten_img = 1/np.absolute(np.fft.fftshift(np.fft.fft2(img)))
                elif whiten.isnumeric():
                    whiten_img = exponential_distance(*img.shape, (img.shape[0]-img.shape[0]%2)/2.0, (img.shape[1]-img.shape[1]%2)/2.0, whiten) # whitening exponent is the additive inverse of the image exponent
                elif whiten.shape == img.shape:
                    whiten_img = whiten
                else:
                    raise ValueError('Unable to interpret the whitening parameter.')
                
                whiten_tar = cut_center(whiten_img, tar)
                
                if weight.shape == img.shape:
                    weight_img = weight # Weighting matrix is the multiplicative inverse of the local variance matrix of the image.
                elif weight.isnumeric():
                    weight_img = 1/local_std(img, weight)
                else:
                    raise ValueError('Unable to interpret the weighting parameter.')
                    
                weight_tar = cut_center(weight_img, tar)
                
                whitened_weighted_img = filter_fft(img*weight_img, whiten_img) 
                whitened_weighted_tar = filter_fft(tar*weight_tar, whiten_tar)
                
                size = uncer['size']
                dist = distance_to_point(*img.shape, (img.shape[0]-img.shape[0]%2)/2,(img.shape[1]-img.shape[1]%2)/2)
                circ = circ_mask(*img.shape, size)
                uncer_mat = uncer['func'](dist) * circ
                uncer_mat /= np.einsum('ij->', uncer_mat)
                
                if uncer['info'] == "sample":
                    responses = np.zeros_like(img)
                    rand_mat = np.random.rand(*img.shape)
                    for i in np.ndindex(img):
                        if uncer_mat[i] > rand_mat[i]:
                            responses[i]=self.inner_prod(whitened_weighted_img, whitened_weighted_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                elif uncer['info'] == "prior":
                    likely = np.zeros_like(img)
                    for i in np.ndindex(img):
                        if circ[i] == 1:
                            likely[i] = self.inner_prod(whitened_weighted_img, whitened_weighted_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                    responses = uncer_mat*likely
                    responses[responses>0] = np.log(responses[responses>0])
                    
                return responses, whitened_weighted_img, whitened_weighted_tar
                
            self.respond = lambda img, tar: func(img, tar, whiten, weight, uncer)
        elif self.method == 'UETM':
            def func(img, tar, csf, uncer):
                csfed_img = filter_fft(img, csf_filter(*img.shape,**csf))
                csfed_tar = filter_fft(tar, csf_filter(*tar.shape, **csf))
                
                size = uncer['size']
                dist = distance_to_point(*img.shape, (img.shape[0]-img.shape[0]%2)/2,(img.shape[1]-img.shape[1]%2)/2)
                circ = circ_mask(*img.shape, size)
                uncer_mat = uncer['func'](dist) * circ
                uncer_mat /= np.einsum('ij->', uncer_mat)
                
                if uncer['info'] == "sample":
                    responses = np.zeros_like(img)
                    rand_mat = np.random.rand(*img.shape)
                    for i in np.ndindex(img):
                        if uncer_mat[i] > rand_mat[i]:
                            responses[i]=self.inner_prod(csfed_img, csfed_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                elif uncer['info'] == "prior":
                    likely = np.zeros_like(img)
                    for i in np.ndindex(img):
                        if circ[i] == 1:
                            likely[i] = self.inner_prod(csfed_img, csfed_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                    responses = uncer_mat*likely
                    responses[responses>0] = np.log(responses[responses>0])
                    
                return responses, csfed_img, csfed_tar
            
            self.respond = lambda img, tar: func(img, tar, csf, uncer)
        elif self.method == 'URETM':
            def func(img, tar, weight, csf, uncer):
                csfed_img = filter_fft(img, csf_filter(*img.shape,**csf))
                csfed_tar = filter_fft(tar, csf_filter(*tar.shape, **csf))
                
                if weight.shape == img.shape:
                    weight_img = weight # Weighting matrix is the multiplicative inverse of the local standard deviation matrix of the image.
                elif weight.isnumeric():
                    weight_img = 1/local_std(csfed_img, weight)
                else:
                    raise ValueError('Unable to interpret the weighting parameter.')
                
                weight_tar = cut_center(weight_img, tar)
                
                weighted_csfed_img = csfed_img*weight_img
                weighted_csfed_tar = csfed_tar*weight_tar
                
                size = uncer['size']
                dist = distance_to_point(*img.shape, (img.shape[0]-img.shape[0]%2)/2,(img.shape[1]-img.shape[1]%2)/2)
                circ = circ_mask(*img.shape, size)
                uncer_mat = uncer['func'](dist) * circ
                uncer_mat /= np.einsum('ij->', uncer_mat)
            
                if uncer['info'] == "sample":
                    responses = np.zeros_like(img)
                    rand_mat = np.random.rand(*img.shape)
                    for i in np.ndindex(img):
                        if uncer_mat[i] > rand_mat[i]:
                            responses[i]=self.inner_prod(weighted_csfed_img, weighted_csfed_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                elif uncer['info'] == "prior":
                    likely = np.zeros_like(img)
                    for i in np.ndindex(img):
                        if circ[i] == 1:
                            likely[i] = self.inner_prod(weighted_csfed_img, weighted_csfed_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                    responses = uncer_mat*likely
                    responses[responses>0] = np.log(responses[responses>0])
                    
                return responses, weighted_csfed_img, weighted_csfed_tar
            
            self.respond = lambda img, tar: func(img, tar, weight, csf, uncer)
        elif self.method == 'UERTM':
            def func(img, tar, weight, csf, uncer):
                if weight.shape == img.shape:
                    weight_img = weight # Weighting matrix is the multiplicative inverse of the local standard deviation matrix of the image.
                elif weight.isnumeric():
                    weight_img = 1/local_std(img, weight)
                else:
                    raise ValueError('Unable to interpret the weighting parameter.')
                    
                weight_tar = cut_center(weight_img, tar)
                
                csfed_weighted_img = filter_fft(img*weight_img, csf_filter(*img.shape,**csf))
                csfed_weighted_tar = filter_fft(tar*weight_tar, csf_filter(*tar.shape,**csf))
                
                size = uncer['size']
                dist = distance_to_point(*img.shape, (img.shape[0]-img.shape[0]%2)/2,(img.shape[1]-img.shape[1]%2)/2)
                circ = circ_mask(*img.shape, size)
                uncer_mat = uncer['func'](dist) * circ
                uncer_mat /= np.einsum('ij->', uncer_mat)
            
                if uncer['info'] == "sample":
                    responses = np.zeros_like(img)
                    rand_mat = np.random.rand(*img.shape)
                    for i in np.ndindex(img):
                        if uncer_mat[i] > rand_mat[i]:
                            responses[i]=self.inner_prod(csfed_weighted_img, csfed_weighted_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                elif uncer['info'] == "prior":
                    likely = np.zeros_like(img)
                    for i in np.ndindex(img):
                        if circ[i] == 1:
                            likely[i] = self.inner_prod(csfed_weighted_img, csfed_weighted_tar, i[0]-(tar.shape[0]-tar.shape[0]%2)/2, i[1]-(tar.shape[1]-tar.shape[1]%2)/2)
                    responses = uncer_mat*likely
                    responses[responses>0] = np.log(responses[responses>0])
                    
                return responses, csfed_weighted_img, csfed_weighted_tar
            
            self.respond = lambda img, tar: func(img, tar, weight, csf, uncer)
        elif self.method == 'UDIY':
            self.respond = resp_func # return responses, filtered image, and filtered target
            