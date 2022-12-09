#! /usr/bin/env python
# torus: https://en.wikipedia.org/wiki/Torus

import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator
from .utils import NumpyDataset
from scipy.special import i0

logger = logging.getLogger(__name__)


class SwissRollSimulator(BaseSimulator):
    def __init__(self, latent_dim=2, data_dim=3, epsilon=0., latent_distribution='correlated', noise_type=None):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._epsilon = epsilon
        
        self._latent_distribution = latent_distribution
        self._noise_type = noise_type
        assert data_dim > latent_dim
        
    def latent_dist(self):
        return self._latent_distribution
    
    def manifold(self):
        return 'swiss_roll'
    
    def dataset(self):
        return 'swiss_roll'+'_'+self._latente_distribution

    def data_dim(self):
        return self._data_dim

    def latent_dim(self):
        return self._latent_dim

    def sample(self, n, parameters=None,mode='numpy'):
        u, v = self._draw_z(n)
        x = self._transform_z_to_x(u,v,mode='numpy')
        return x
    
    def sample_and_noise(self,n,sig2 = 0.0, mode='numpy'):
        # import pdb
        # pdb.set_trace()
        u, v = self._draw_z(n)
        x = self._transform_z_to_x(u,v,mode='numpy')
        noise = self.create_noise(np.array(x,copy=True),u,v,sig2)
        return np.stack([x, noise],axis=-1)   #create new dataset for loader

    def create_noise(self,x,u,v,sig2):
        if self._noise_type == 'gaussian':
            noise = np.sqrt(sig2) * np.random.randn(*x.shape)
        else: noise = np.zeros(*x.shape)
        
        return noise      

    def _draw_z(self, n):
        if self._latent_distribution == 'mixture':
            n_samples = np.random.multinomial(n,[1/3]*3,size=1)
            n_samples_1 = n_samples[0,0]
            n_samples_2 = n_samples[0,1]
            n_samples_3 = n_samples[0,2]
            
            kappa, mu11, mu12, mu21, mu22, mu31, mu32 = 2, 0.1, 0.1, 0.5, 0.8, 0.8, 0.8 
            
            u_1 = self._translate_inverse(np.random.vonmises( self._translate(mu11) ,kappa,n_samples_1))
            v_1 = self._translate_inverse(np.random.vonmises( self._translate(mu12) ,kappa,n_samples_1))
            u_2 = self._translate_inverse(np.random.vonmises( self._translate(mu21) ,kappa,n_samples_2))
            v_2 = self._translate_inverse(np.random.vonmises( self._translate(mu22) ,kappa,n_samples_2))
            u_3 = self._translate_inverse(np.random.vonmises( self._translate(mu31) ,kappa,n_samples_3))
            v_3 = self._translate_inverse(np.random.vonmises( self._translate(mu32) ,kappa,n_samples_3))
            
            u = np.concatenate([u_1,u_2,u_3],axis=0)
            v = np.concatenate([v_1,v_2,v_3],axis=0)
            
        elif self._latent_distribution == 'unimodal':
            mu1, mu2, kappa = 0.5, 0.8, 2 
            u = self._translate_inverse( np.random.vonmises( self._translate(mu1) ,kappa,n ) )  #np.random.vonmises(-np.pi + 2*np.pi*mu1,kappa,n )+np.pi) / (2*np.pi)
            v = self._translate_inverse( np.random.vonmises( self._translate(mu2) ,kappa,n ) )      
            
        elif self._latent_distribution == 'correlated':
            kappa, mu = 5, 0.0
            u = np.random.uniform(0,1,n)                      #(np.random.vonmises(0,0,n )+np.pi) / (2*np.pi)
            v = self._translate_inverse(np.random.vonmises(self._translate(u),kappa,n)) 
        
        elif self._latent_distribution == 'uniform':
            u = np.random.uniform(0,1,n)
            v = np.random.uniform(0,1,n)   

        return u, v
    
    
    def _transform_z_to_x(self, u_, v_, mode='train'):
        u = u_.reshape([1,-1])
        v = v_.reshape([1,-1])
        
        t = 1.5 * np.pi * (1 + 2 * v)
        x1 = t * np.cos(t)
        x2 = 21 * u  
        x3 = t * np.sin(t)
        
        data = np.concatenate([x1, x2, x3])
        data=data.T
        if mode=='train':            
            params = np.ones(data.shape[0])
            data = NumpyDataset(data, params)
        return data

    
