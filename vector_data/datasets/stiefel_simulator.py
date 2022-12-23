#! /usr/bin/env python


import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator
from .utils import NumpyDataset
from scipy.special import i0
import torch

logger = logging.getLogger(__name__)


class StiefelSimulator(BaseSimulator):
    def __init__(self, latent_dim=1, data_dim=4, kappa=6.0, epsilon=0.,latent_distribution='mixture',noise_type='gaussian'):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._epsilon = epsilon
        
        self._latent_distribution = latent_distribution
        self._noise_type = noise_type

        self.kappa = 6.0
        self.mu1 = 0
        self.mu2 = -np.pi/2            
        self.mu3 = np.pi/2 
        self.mu4 = np.pi          
        
        assert data_dim > latent_dim
        
    def latent_dist(self):
        return self._latent_distribution
    
    def manifold(self):
        return 'stiefel'
    
    def dataset(self):
        return 'stiefel'+'_'+self._latente_distribution

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
        theta = self._draw_z(n)
        x = self._transform_z_to_x(theta,mode='numpy')
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = self._epsilon * np.random.normal(size=(n, 2))
        return x + noise

    def sample_and_noise(self,n,parameters=None, sig2 = 0.0, mode='numpy'):
        theta = self._draw_z(n)
        x = self._transform_z_to_x(theta,mode='numpy')
        noise = self.create_noise(x,theta,sig2)
        return np.stack([x, noise],axis=-1)   

    def create_noise(self,x,theta,sig2):
        if self._noise_type == 'gaussian':
            noise = np.sqrt(sig2) * np.random.randn(*x.shape)
        elif self._noise_type == 'normal': 
            # print('x',x.shape)
            n1 = x / np.sqrt(2)
            n2 = np.zeros(x.shape)
            n2[:,1:3] = 1 / np.sqrt(2)
            n3 = np.zeros(x.shape)
            n3[:,0] = 1 / np.sqrt(2)
            n3[:,3] = -1 / np.sqrt(2)
            
            noise  =  np.sqrt(sig2) * np.random.randn(len(theta),1) * n1+np.sqrt(sig2) * np.random.randn(len(theta),1) * n2+np.sqrt(sig2) * np.random.randn(len(theta),1) * n3 
                     
        else: noise = np.zeros(x.shape)
        return noise  

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def _draw_z(self, n):
        if self._latent_distribution == 'mixture':
            n_samples = np.random.multinomial(n,[1/4]*4,size=1)
            n_samples_1 = n_samples[0,0]
            n_samples_2 = n_samples[0,1]
            n_samples_3 = n_samples[0,2]
            n_samples_4 = n_samples[0,3]
            
            theta1 = np.random.vonmises(self.mu1,self.kappa,n_samples_1 )
            theta2 = np.random.vonmises(self.mu2,self.kappa,n_samples_2 )
            theta3 = np.random.vonmises(self.mu3,self.kappa,n_samples_3 )
            theta4 = np.random.vonmises(self.mu4,self.kappa,n_samples_4 )
            
            theta = np.concatenate([theta1,theta2,theta3,theta4],axis=0)
        return theta
    
    
    def _transform_z_to_x(self, theta, mode='train'):
        x1 = np.cos(theta)
        x2 = np.sin(theta)
        sign =  1 #(-1)**np.random.binomial(size=len(theta), n=1, p= 0.5)
        x = np.stack([x1,x2,-sign*x2,sign*x1], axis=1) 
        if mode=='train':
            x = NumpyDataset(x, params)
        return x


    def _transform_x_to_z(self, x):
        raise NotImplementedError

    def _density(self, theta_):
        
        if self._latent_distribution == 'mixture':
            probs = 1/4* (np.exp(self.kappa*np.cos(theta_-self.mu1)) + np.exp(self.kappa*np.cos(theta_-self.mu2))
                         +np.exp(self.kappa*np.cos(theta_-self.mu3)) + np.exp(self.kappa*np.cos(theta_-self.mu4))
                         ) * (1/(2*np.pi*i0(self.kappa)))                
        return probs
    
    def generate_grid(self,n,mode='sphere'):
        theta = np.linspace(-np.pi, np.pi, n)
        latent = theta
        
        true_probs = self._density(theta)
        if mode == 'data_space':            
            data = self._transform_z_to_x(theta,mode='test')
        else: data = latent
        jacobians = 2
        
        return data, latent, true_probs, jacobians, 1
    
    def calculate_sigma_bound(self,theta_, v_=None):     
        def circle_term(z,alpha):
            return self.kappa *  (self.kappa*np.sin(z-alpha)**2 - np.cos(z-alpha) )
       
        def circle_term_phi(z,alpha):
            return self.kappa * (self.kappa*np.sin(z-alpha)**2 - np.cos(z-alpha) )
        
        def jacobian(theta):                   
            return np.sin(theta)
        
        def unimodal(theta,alpha):
            return (1/4)*np.exp( self.kappa*np.cos(theta-alpha) ) * (1/(2*np.pi*i0(self.kappa)))
       
        if self._latent_distribution == 'mixture':
            bound = 2 * self._density(theta_) / (  unimodal(theta_,self.mu1) * circle_term(theta_,self.mu1) / 2
                                                  +unimodal(theta_,self.mu2) * circle_term(theta_,self.mu2) / 2
                                                  +unimodal(theta_,self.mu3) * circle_term(theta_,self.mu3) / 2
                                                  +unimodal(theta_,self.mu4) * circle_term(theta_,self.mu4) / 2
                                                 )
        return bound
    
    def calculate_gauss_curvature(self, z1, z2=None):        
        return 1 * np.ones(*z1.shape)
    