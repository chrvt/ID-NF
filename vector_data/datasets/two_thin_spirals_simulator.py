#! /usr/bin/env python
# torus: https://en.wikipedia.org/wiki/Torus

import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator
from .utils import NumpyDataset
from scipy.special import i0

logger = logging.getLogger(__name__)


class TwoThinSpiralsSimulator(BaseSimulator):
    def __init__(self, latent_dim=1, data_dim=2, epsilon=0., latent_distribution='exponential', noise_type=None):
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
        return '2thin_spirals'
    
    def dataset(self):
        return '2thin_spirals'+'_'+self._latente_distribution

    def is_image(self):
        return False

    def data_dim(self):
        return self._data_dim

    def latent_dim(self):
        return self._latent_dim

    def parameter_dim(self):
        return None

    def log_density(self, x, parameters=None, precise=False):
        raise NotImplementedError

    def sample(self, n, parameters=None,mode='numpy'):
        z1, z2 = self._draw_z(n)
        x_1 = self._transform_z_to_x(z1,mode='numpy')
        x_2 = -1 * self._transform_z_to_x(z2,mode='numpy')
        x = np.concatenate([x_1,x_2])
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = np.sqrt(self._epsilon) * np.random.normal(size=(n, 2))
        return x + noise
    
    def sample_and_noise(self,n,sig2 = 0.0, mode='numpy'):
        z1, z2 = self._draw_z(n)
        x_1 = self._transform_z_to_x(z1,mode='numpy')
        x_2 = -1 * self._transform_z_to_x(z2,mode='numpy')
        x = np.concatenate([x_1,x_2])
        # import pdb
        # pdb.set_trace()
        noise = self.create_noise(np.array(x,copy=True),sig2)
        return np.stack([x, noise],axis=-1)   #create new dataset for loader

    def create_noise(self,x,sig2):
        if self._noise_type == 'gaussian':
            noise = np.sqrt(sig2) * np.random.randn(*x.shape)
        elif self._noise_type == 'normal':
            normal_ = self._transform_x_to_normal(x)
            noise = np.sqrt(sig2) * np.random.randn(len(u),1) * (normal_ / np.linalg.norm(normal_ ,axis=1).reshape(normal_.shape[0],1) ) ## (normal_ / np.linalg.norm(normal_ ,axis=1).reshape(normal_.shape[0],1) )
            # import pdb
            # pdb.set_trace()
        else: noise = np.zeros(*x.shape)
        
        return noise      

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def _draw_z(self, n):
        if self._latent_distribution == 'exponential':
            latent_1 = np.random.exponential(scale=0.3,size=int(n/2))
            latent_2 = np.random.exponential(scale=0.3,size=int(n/2))
            
        # elif self._latent_distribution == 'log_normal':
        #     s = 0.5
        #     mu = 0
        #     latent_ = np.random.lognormal(size=n,mean=mu,sigma=s)
        
        return latent_1[::-1], latent_2
    
    
    def _transform_z_to_x(self, latent_, mode='train'):
        latent = np.sqrt(latent_ ) * 540 * (2 * np.pi) / 360
        
        d1x = - np.cos(latent) * latent 
        d1y =   np.sin(latent) * latent 
        data = np.stack([ d1x,  d1y], axis=1) / 3
        if mode=='train':            
            params = np.ones(data.shape[0])
            data = NumpyDataset(data, params)
        return data

    def _transform_x_to_normal(self,data):
        norm = np.linalg.norm(data,axis=1).reshape([data.shape[0],1])
        z = 3 * norm
        e_r = data / norm
        R = np.array(([0,-1],[+1,0])) 
        e_phi = +1*np.matmul(e_r,R)
        x_norm = (e_r + z * e_phi)/3 
        normal = np.matmul(x_norm,R)
        return normal
    
    def _transform_x_to_z(self, x):
        raise NotImplementedError

    def _density(self, z_):
        if self._latent_distribution == 'exponential':
            from scipy.stats import expon
            probs= expon.pdf(z_,scale=0.3) 
            
        # elif self._latent_distribution == 'log_normal':
        #     s = 0.5
        #     mu = 0
        #     from scipy.stats import lognorm
        #     probs= lognorm.pdf(latent_test,loc=mu,s=s) 
            
        return probs
    
    def generate_grid(self,n,mode='data_space'):
        multiplier = 1
        z_1 = np.linspace(-2.5,0, int(n/2)+1)[:-1]
        z_2 = np.linspace(0, 2.5, int(n/2)+1)[1:]
        # z = np.sqrt(z_exp ) * 540 * (2 * np.pi) / 360
        latent = np.concatenate([z_1 ,z_2])
        # xx, yy = np.meshgrid(z, z)
        # grid = np.stack((xx.flatten(), yy.flatten()), axis=1)
        true_probs = np.concatenate( [self._density(-1*z_1),self._density(z_2)] )
        if mode == 'data_space':            
            data = np.concatenate( [-1*self._transform_z_to_x(-1*z_1,mode='test'), self._transform_z_to_x(z_2,mode='test')] )
        else: data = None
        c1 = 540 * 2* np.pi / 360
        r1 = np.sqrt(-1*z_1) * c1
        r2 = np.sqrt(z_2) * c1
        jacobians = np.concatenate( [((1+r1**2)/r1**2), ((1+r2**2)/r2**2)] )* c1**4 / 36
        # import pdb
        # pdb.set_trace()
        return data, latent, true_probs, jacobians, multiplier #torch.tensor(data)
        


