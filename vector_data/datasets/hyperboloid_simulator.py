#! /usr/bin/env python
# torus: https://en.wikipedia.org/wiki/Torus

import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator
from .utils import NumpyDataset
# from .utils import _cartesian_to_spherical
from scipy.special import i0

logger = logging.getLogger(__name__)


class HyperboloidSimulator(BaseSimulator):
    def __init__(self, latent_dim=2, data_dim=3, a=1, b=1 , c=1, epsilon=0., latent_distribution='correlated', noise_type=None):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._epsilon = epsilon
        
        self._latent_distribution = latent_distribution
        self._noise_type = noise_type
        
        self._a = a 
        self._b = b
        self._c = c 
            
        assert data_dim > latent_dim
        
    def latent_dist(self):
        return self._latent_distribution
    
    def manifold(self):
        return 'hyperboloid'
    
    def dataset(self):
        return 'hyperboloid'+'_'+self._latente_distribution

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
        v, theta = self._draw_z(n)
        x = self._transform_z_to_x(v,theta,mode='numpy')
        # z_hat =  self._transform_x_to_z(x)
        # import pdb
        # pdb.set_trace()
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = self._epsilon * np.random.normal(size=(n, 2))
        return x + noise

    def sample_and_noise(self,n,parameters=None, sig2 = 0.0, mode='numpy'):
        v, theta = self._draw_z(n)
        x = self._transform_z_to_x(v,theta,mode='numpy')
        noise = self.create_noise(x,v,theta,sig2)
        
        return np.stack([x, noise],axis=-1)   
    
    def create_noise(self,x,v,theta,sig2):
        if self._noise_type == 'gaussian':
            noise = np.sqrt(sig2) * np.random.randn(*x.shape)
        elif self._noise_type == 'normal':
            normal_ = self._transform_z_to_x(v,theta,sign=-1)
            noise = np.sqrt(sig2) * np.random.randn(len(v),1) * (normal_ / np.linalg.norm(normal_ ,axis=1).reshape(normal_.shape[0],1) )
        else: noise = np.zeros(*x.shape)
        return noise         
            

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def _draw_z(self, n):
        if self._latent_distribution == 'mixture':
            n_ = np.random.multinomial(n,[1/2]*2,size=1)
            n_1 = n_[0,0]
            n_2 = n_[0,1]
            
            kappa, mu11, mu12  = 6, -np.pi/2, np.pi/2
            theta_smpl_1 = (np.random.vonmises(mu11,kappa,n_1)) + np.pi
            theta_smpl_2 = (np.random.vonmises(mu12,kappa,n_2)) + np.pi
            theta = np.concatenate([theta_smpl_1,theta_smpl_2],axis=0)
            
            n_ = np.random.multinomial(n,[1/1]*1,size=1)
            v   = np.random.uniform(1.0,1.5,n_[0])            
            
        elif self._latent_distribution == 'correlated':
            kappa = 6
            v     = np.random.uniform(0,2.0,n)
            theta = (np.random.vonmises(v,kappa,n)+np.pi) 
        
        elif self._latent_distribution == 'unimodal':
            kappa, mu1, mu2  = 6, np.pi/2, np.pi
            theta =  (np.random.vonmises(0,0,n)+np.pi)  #(np.random.vonmises(0,0,n_samples)+np.pi) #/2 #(2*np.pi)
            v     = np.random.exponential(scale=0.5, size= theta.shape )#1*np.random.randn(*theta.shape)
        
        return v, theta
    
    
    def _transform_z_to_x(self, v, theta, sign=1, mode='numpy'):
        x1 = self._a*(np.sinh(v))*np.cos(theta)
        x2 = self._b*(np.sinh(v))*np.sin(theta)
        x3 = sign*self._c*np.cosh(v)
        x = np.stack([x3,x2,x1], axis=1) 
        if mode=='train':            
            params = np.ones(x.shape[0])
            x = NumpyDataset(x, params)
        return x
    
    # def _transform_z_to_sphere(self, theta,phi):
    #     d1x = np.cos(theta) * np.cos(phi)
    #     d1y = np.cos(theta) * np.sin(phi)
    #     d1z = np.sin(theta)
    #     x = np.stack([ d1x, d1y, d1z], axis=1) 
    #     return x

    def _transform_x_to_z(self, x):
        raise NotImplementedError
        # bs = x.shape[0]
        # x3 = x[:,0]
        # v_ = np.log(x3+np.sqrt(x3**2-1))
        # x12 = x[:,[2,1]]
        # sinh_v = np.sinh(v_).reshape([bs,1])
        # theta = _cartesian_to_spherical(x12/sinh_v)
        # # theta = np.arccos(x12 / sinh_v) 
        # latent = np.concatenate([v_.reshape([bs,1]),theta.reshape([bs,1])],axis=1)
        
        # # import pdb
        # # pdb.set_trace()
        
        # return latent

    def _density(self,v,theta):
        if self._latent_distribution == 'mixture':
            kappa, mu11, mu12  = 6, np.pi/2, 3*np.pi/2
            probs_theta = 0.5*( np.exp(kappa*np.cos((theta-mu11))) + np.exp(kappa*np.cos(theta-mu12)) ) *(1/(2*np.pi*i0(kappa)))
            probs_v = np.zeros(*v.shape)
            for i in range(len(v)):
                if 1.0<=v[i] <= 1.5:
                    probs_v[i]=2
            probs = probs_v * probs_theta 
        elif self._latent_distribution == 'correlated':
            kappa = 6
            probs_theta = 1*( np.exp(kappa*np.cos((theta-v-np.pi)))) *(1/(2*np.pi*i0(kappa)))
            #1*( np.exp(kappa*np.cos((theta-v+np.pi)))) *(1/(2*np.pi*i0(kappa)))
            probs_v = np.ones(*v.shape)/2
            probs = probs_v * probs_theta 
        elif self._latent_distribution == 'unimodal':
            sig2 = 1 
            probs_theta = 1/(2*np.pi)
            probs_v =  2*np.exp(-2*v) #np.sqrt(2*np.pi*sig2)*np.exp(-v**2/(2*sig2))
            probs = probs_v * probs_theta             
        return probs
    
    def generate_grid(self,n,mode='data_space'):
        theta = np.linspace(0, 2*np.pi, n+1)[1:n+1]  #not take the same point twice
        v = np.linspace(0, 2, n+1)[1:] 
        # v = np.linspace(0, 2, n+1)[1:]
        latent = [v,theta]
        xx, yy = np.meshgrid(v, theta)
        
        grid = np.stack((xx.flatten(), yy.flatten()), axis=1)
        true_probs = self._density(grid[:,0],grid[:,1])
        if mode == 'data_space':            
            data = self._transform_z_to_x(grid[:,0],grid[:,1],mode='test')
        else: data = [xx, yy]
        jacobians = np.sqrt( ( self._a**2 * np.sinh(xx)**2 + self._c**2 * np.cosh(xx)**2 ) * self._a**2 * np.sinh(xx)**2 )
        # jacobians[:,0] = 1
        # import pdb
        # pdb.set_trace()
        return data, latent, true_probs, jacobians, 1 #torch.tensor(data)
        
    def calculate_sigma_bound(self,v,theta):
        if self._latent_distribution == 'correlated':
            bound = 2/( (36*np.sin(theta-v-np.pi)**2 + 6*np.cos(theta-v-np.pi)) * ( 1 / (np.cosh(v)**2+np.sinh(v)**2) + 1/np.sinh(v)**2 ) )
        elif self._latent_distribution == 'unimodal':
            bound = (np.cosh(v)**2+np.sinh(v)**2) / 4  
        else: bound = 0
        return bound

    def calculate_gauss_curvature(self,v,theta): 
        #https://mathworld.wolfram.com/Two-SheetedHyperboloid.html
        z = self._c*np.cosh(v)
        return self._c**6 / (np.abs(self._c**4-(self._a**2+self._c**2)*z**2))