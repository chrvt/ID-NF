#! /usr/bin/env python
# torus: https://en.wikipedia.org/wiki/Torus

import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator
from .utils import NumpyDataset
from scipy.special import i0

logger = logging.getLogger(__name__)


class TorusSimulator(BaseSimulator):
    def __init__(self, latent_dim=2, data_dim=3, kappa=2, c=1 , a=0.6, epsilon=0., latent_distribution='correlated', noise_type=None):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._epsilon = epsilon
        
        self._latent_distribution = latent_distribution
        self._noise_type = noise_type
        
        self._kappa = kappa #concentration
        self._c = c
        self._a = a      
        self._mu = 1.94
            
        assert data_dim > latent_dim
        
    def latent_dist(self):
        return self._latent_distribution
    
    def manifold(self):
        return 'torus'
    
    def dataset(self):
        return 'torus'+'_'+self._latente_distribution

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
        theta, phi = self._draw_z(n)
        x = self._transform_z_to_x(theta,phi,mode='numpy')
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = self._epsilon * np.random.normal(size=(n, 2))
        return x + noise

    def sample_and_noise(self,n,sig2 = 0.0, mode='numpy'):
        theta, phi = self._draw_z(n)
        x = self._transform_z_to_x(theta,phi,mode='numpy')
        noise = self.create_noise(x,theta,phi,sig2)
        return np.stack([x, noise],axis=-1)   
    
    def create_noise(self,x,theta,phi,sig2):
        if self._noise_type == 'gaussian':
            noise = np.sqrt(sig2) * np.random.randn(*x.shape)
        elif self._noise_type == 'normal':
            noise_ = self._transform_z_to_sphere(theta,phi)
            noise = np.sqrt(sig2) * np.random.randn(len(theta),1) * noise_
        else: 
            noise = np.zeros(*x.shape)
        return noise         
            

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def _draw_z(self, n):
        if self._latent_distribution == 'mixture':
            n_samples = np.random.multinomial(n,[1/3]*3,size=1)
            n_samples_1 = n_samples[0,0]
            n_samples_2 = n_samples[0,1]
            n_samples_3 = n_samples[0,2]
            
            kappa, mu11, mu12, mu21, mu22, mu31, mu32 = self._kappa, 0.21, 2.85, 1.89, 6.18, 3.77, 1.56 
            
            theta1 = np.random.vonmises(mu11,kappa,n_samples_1 )
            phi1 = np.random.vonmises(mu12,kappa,n_samples_1 )
            theta2 = np.random.vonmises(mu21,kappa,n_samples_2 )
            phi2 = np.random.vonmises(mu22,kappa,n_samples_2 )
            theta3 = np.random.vonmises(mu31,kappa,n_samples_3)
            phi3 = np.random.vonmises(mu32,kappa,n_samples_3 )
            
            theta = np.concatenate([theta1,theta2,theta3],axis=0)
            phi = np.concatenate([phi1,phi2,phi3],axis=0)  
            
        elif self._latent_distribution == 'correlated':
            phi = np.random.vonmises(0,0,n) 
            theta = np.random.vonmises((self._mu-phi),self._kappa,n)
        
        return theta, phi
    
    
    def _transform_z_to_x(self, theta,phi, mode='train'):
        d1x = (self._c + self._a*np.cos(theta)) * np.cos(phi)
        d1y = (self._c + self._a*np.cos(theta)) * np.sin(phi)
        d1z = (self._a * np.sin(theta))
        x = np.stack([ d1x, d1y, d1z], axis=1) 
        if mode=='train':            
            params = np.ones(x.shape[0])
            x = NumpyDataset(x, params)
        return x
    
    def _transform_z_to_sphere(self, theta,phi):
        d1x = np.cos(theta) * np.cos(phi)
        d1y = np.cos(theta) * np.sin(phi)
        d1z = np.sin(theta)
        x = np.stack([ d1x, d1y, d1z], axis=1) 
        return x

    def _transform_x_to_z(self, x):
        raise NotImplementedError

    def _density(self, data):
        theta = data[0]
        phi = data[1]
        if self._latent_distribution == 'mixture':
            kappa, mu11, mu12, mu21, mu22, mu31, mu32 = 2, 0.21, 2.85, 1.89, 6.18, 3.77, 1.56 
            probs =    (1/3)*( np.exp(kappa*np.cos(theta-mu11)+kappa*np.cos(phi-mu12))
                          +  np.exp(kappa*np.cos(theta-mu21)+kappa*np.cos(phi-mu22))
                          +  np.exp(kappa*np.cos(theta-mu31)+kappa*np.cos(phi-mu32)) )*(1/(2*np.pi*i0(kappa))**2) 
        elif self._latent_distribution == 'correlated':
            probs = 1/(2*np.pi) * np.exp(self._kappa*np.cos(phi+theta-self._mu)) *(1/(2*np.pi*i0(self._kappa))**1)   
        elif self._latent_distribution == 'unimodal':
            mu1, mu2, kappa = 4.18-np.pi, 5.96-np.pi, 2
            probs = np.exp(kappa*np.cos(phi-mu1) + kappa*np.cos(theta-mu2))*(1/(2*np.pi*i0(kappa))**2)               
        return probs
    
    def generate_grid(self,n,mode='data_space'):
        theta = np.linspace(-np.pi, np.pi, n) #[1:] #+1  [1:n+1]
        phi = np.linspace(-np.pi, np.pi, n) #+1    [1:n+1]
        latent = [theta,phi]
        xx, yy = np.meshgrid(theta, phi)
        
        grid = np.stack((xx.flatten(), yy.flatten()), axis=1)
        true_probs = self._density([grid[:,0],grid[:,1]])
        if mode == 'data_space':            
            data = self._transform_z_to_x(grid[:,0],grid[:,1],mode='test')
        else: data = [xx, yy]
        jacobians = np.abs(self._a*(self._c+self._a*np.cos(xx)))
        return data, latent, true_probs, jacobians, 1 #torch.tensor(data)    

    def calculate_sigma_bound(self,theta, phi): #theta=v, phi = u
        def circle_term(x_,alpha):
            return self._kappa * (self._kappa*np.sin(x_-alpha)**2 - np.cos(x_-alpha) )
        def jacobian(theta):                   
            return self._c+self._a*np.cos(theta)
        def unimodal(theta,phi,mu,m):
            kappa = 2
            return (1/3)*np.exp(kappa*np.cos(theta-mu)+kappa*np.cos(phi-m))*(1/(2*np.pi*i0(kappa))**2) 
       
        if self._latent_distribution == 'mixture':
            kappa, mu11, mu12, mu21, mu22, mu31, mu32 = 2, 0.21, 2.85, 1.89, 6.18, 3.77, 1.56
            bound = 2 * self._density([theta,phi]) / ( unimodal(theta,phi,mu11,mu12) * (circle_term(theta,mu11)/jacobian(theta)**2 + circle_term(phi,mu12)/self._a**2 )  
                                                      +unimodal(theta,phi,mu21,mu22) * (circle_term(theta,mu21)/jacobian(theta)**2 + circle_term(phi,mu22)/self._a**2 )
                                                      +unimodal(theta,phi,mu31,mu32) * (circle_term(theta,mu31)/jacobian(theta)**2 + circle_term(phi,mu32)/self._a**2 ) )
        
        elif self._latent_distribution == 'correlated':
            bound = (2 * self._a**2 * jacobian(theta)**2 ) / ( circle_term(phi+theta,self._mu) * (self._a**2 + jacobian(theta)**2)  )
       
        return bound
    
    def calculate_gauss_curvature(self,theta,phi):
        return np.cos(theta)/(self._c*(self._a + self._c * np.cos(theta)) )