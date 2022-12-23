#! /usr/bin/env python
# torus: https://en.wikipedia.org/wiki/Torus

import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator
from .utils import NumpyDataset
from scipy.special import i0

logger = logging.getLogger(__name__)


class ThinSpiralSimulator(BaseSimulator):
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
        return 'thin_spiral'
    
    def dataset(self):
        return 'thin_spiral'+'_'+self._latente_distribution

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
        z = self._draw_z(n)
        x = self._transform_z_to_x(z,mode='numpy')
        # z_hat =  self._transform_x_to_z(x)
        # import pdb
        # pdb.set_trace()
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = np.sqrt(self._epsilon) * np.random.normal(size=(n, 2))
        return x + noise
    
    def sample_and_noise(self,n,sig2 = 0.0, mode='numpy'):
        z = self._draw_z(n)
        x = self._transform_z_to_x(z,mode='numpy')
        # import pdb
        # pdb.set_trace()
        noise = self.create_noise(np.array(x,copy=True),z,sig2)
        return np.stack([x, noise],axis=-1)   #create new dataset for loader

    def create_noise(self,x,z,sig2):
        if self._noise_type == 'gaussian':
            noise = np.sqrt(sig2) * np.random.randn(*x.shape)
        elif self._noise_type == 'normal':
            normal_ = self._transform_x_to_normal(x)
            noise = np.sqrt(sig2) * np.random.randn(len(z),1) * (normal_ / np.linalg.norm(normal_ ,axis=1).reshape(normal_.shape[0],1) ) ## (normal_ / np.linalg.norm(normal_ ,axis=1).reshape(normal_.shape[0],1) )
            # import pdb
            # pdb.set_trace()
        else: noise = np.zeros(*x.shape)
        
        return noise      

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def _draw_z(self, n):
        if self._latent_distribution == 'exponential':
            latent_ = np.random.exponential(scale=0.3,size=n)
            
        elif self._latent_distribution == 'log_normal':
            s = 0.5
            mu = 0
            latent_ = np.random.lognormal(size=n,mean=mu,sigma=s)
        
        return latent_
    
    
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
        z = ((np.linalg.norm(x,axis=1)/(np.pi))**2).reshape([x.shape[0],1])
        return z

    def _density(self, z_):
        if self._latent_distribution == 'exponential':
            from scipy.stats import expon
            probs= expon.pdf(z_,scale=0.3) 
            
        elif self._latent_distribution == 'log_normal':
            s = 0.5
            mu = 0
            from scipy.stats import lognorm
            probs= lognorm.pdf(z_,loc=mu,s=s) 
            
        return probs
    
    def generate_grid(self,n,mode='data_space'):
        multiplier = 1
        z_ = np.linspace(0, 2.5, n+1)[1:]
        # z = np.sqrt(z_exp ) * 540 * (2 * np.pi) / 360
        latent = z_ #[,np.array([0,1])]
        
        # xx, yy = np.meshgrid(z, z)
        # grid = np.stack((xx.flatten(), yy.flatten()), axis=1)
        
        true_probs = self._density(z_)
        if mode == 'data_space':            
            data = self._transform_z_to_x(z_,mode='test')
        else: data = None
        c1 = 3 * np.pi 
        r = np.sqrt(z_) * c1
        jacobians = ((1+r**2)/r**2) * c1**4 / 36   
        # import pdb
        # pdb.set_trace()
        return data, latent, true_probs, jacobians, multiplier #torch.tensor(data)
        
    def calculate_sigma_bound(self,z,z2=None):
        if self._latent_distribution == 'exponential':
            c1 = 3 * np.pi 
            r = np.sqrt(z) * c1
            jacobians = ((1+r**2)/r**2) * c1**4 / 36 
            bound = 2/(0.3)**2 * jacobians  #2*(1+(3*np.pi*np.sqrt(z))**2) / ( 0.3**2 * (3*np.pi*np.sqrt(z))**2 *2.25*np.pi**4  )
        return bound
    
    def calculate_gauss_curvature(self,z_,z2=None): 
        #https://www.wolframalpha.com/input/?i2d=true&i=+norm%5C%2840%29+D%5B%5C%2840%29-cos%5C%2840%29a*Sqrt%5Bz%5D%5C%2841%29%5C%2841%29*a*Sqrt%5Bz%5D%5C%2844%29sin%5C%2840%29a*Sqrt%5Bz%5D%5C%2841%29*a*Sqrt%5Bz%5D%5C%2841%29%5C%2841%29%2C%7Bz%2C2%7D%5D+%5C%2841%29
        #https://www.wolframalpha.com/input/?i2d=true&i=D%5B%5C%2840%29-cos%5C%2840%29a*Sqrt%5Bz%5D%5C%2841%29%5C%2841%29*a*Sqrt%5Bz%5D%5C%2844%29sin%5C%2840%29a*Sqrt%5Bz%5D%5C%2841%29*a*Sqrt%5Bz%5D%5C%2841%29%5C%2841%29%2Cz%5D
        #https://www.wolframalpha.com/input/?i2d=true&i=+%5C%2840%29+D%5B%5C%2840%29-cos%5C%2840%29a*Sqrt%5Bz%5D%5C%2841%29%5C%2841%29*a*Sqrt%5Bz%5D%5C%2844%29sin%5C%2840%29a*Sqrt%5Bz%5D%5C%2841%29*a*Sqrt%5Bz%5D%5C%2841%29%5C%2841%29%2C%7Bz%2C2%7D%5D+%5C%2841%29
        #https://mathepedia.de/Kruemmung.html 
        a = 3 * np.pi 
        y = np.sqrt(z_ ) * a
        dz1 = 0.5 * a**2 *( np.sin(y)-np.cos(y)/y )
        dz2 = 0.5 * a**2 *( np.cos(y)-np.sin(y)/y )
        ddz1 = (a**4 / (4*y**3)) * ( y*np.sin(y) + (y**2 + 1)*np.cos(y) )
        ddz2 = (a**4 / (4*y**3)) * ( y*np.cos(y) - (y**2 + 1)*np.sin(y) )
        gauss = (dz1*ddz2-ddz1*dz2) / (dz1**2 + dz2**2)**(3/2)
        
        # h1 = 4*z_**(3/2)
        # first = - const * np.cos(z) / h1 - const**2 * np.sin(z)/(2*z_) + z * (  const*np.sin(z) / h1 - const**2 * np.cos(z) / (4*z_)) 
        # second= - const * np.sin(z) / h1 + const**2 * np.cos(z)/(2*z_) + z * ( -const*np.cos(z) / h1 - const**2 * np.sin(z) / (4*z_))
        return  gauss #(first**2 + second**2)