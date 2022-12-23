#! /usr/bin/env python

import numpy as np
from scipy.stats import norm
import logging
#from .base import BaseSimulator
#from .utils import NumpyDataset
from scipy.special import i0
import torch

logger = logging.getLogger(__name__)


class LolipopSimulator(object):
    def __init__(self, latent_dim=1, data_dim=2, kappa=6.0, epsilon=0.,latent_distribution='uniform',noise_type='gaussian'):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._epsilon = epsilon
        
        self._latent_distribution = latent_distribution
        self._noise_type = noise_type

        self.kappa = 6.0
        self.mu11 = 1*np.pi/4       
        self.mu12 = np.pi/2             #northpole
        self.mu21 = 3*np.pi/4 
        self.mu22 = 4*np.pi/3           #southpole
        self.mu31 = 3*np.pi/4 
        self.mu32 = np.pi/2 
        self.mu41 = np.pi/4
        self.mu42 = 4*np.pi/3 
        
        self.first_sample = True

        assert data_dim > latent_dim
        
    def latent_dist(self):
        return self._latent_distribution
    
    def manifold(self):
        return 'lolipop'
    
    def dataset(self):
        return 'lolipop'+'_'+self._latente_distribution

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
        n_disk  = np.random.random_integers(0,n,1)[0]
        n_stick = n - n_disk
        
        phi, r = self._draw_z(n_disk)
        x_disk = self._transform_z_to_disk(phi,r,mode='numpy')
        
        alpha = np.random.random(n_stick) * np.sqrt(2)
        x_stick = np.stack([ alpha, alpha], axis=1) 
        
        x = np.concatenate([x_disk, x_stick]) 
        
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = self._epsilon * np.random.normal(size=(n, 2))
        return x + noise

    def sample_and_noise(self,n,parameters=None, sig2 = 0.0, mode='numpy'):
        n_disk  = np.random.random_integers(0,n,1)[0]
        n_stick = n - n_disk
        
        phi, r = self._draw_z(n_disk)
        x_disk = self._transform_z_to_disk(phi,r,mode='numpy')
        
        alpha = np.random.random(n_stick) * np.sqrt(2)
        x_stick = np.stack([ alpha, alpha], axis=1) 
        
        x = np.concatenate([x_disk, x_stick]) 
        
        noise = self.create_noise(x,phi,r,sig2)
        return np.stack([x, noise],axis=-1)   

    def create_noise(self,x,theta,phi,sig2):
        if self._noise_type == 'gaussian':
            noise = np.sqrt(sig2 + self._epsilon**2) * np.random.randn(*x.shape)
        elif self._noise_type == 'normal': #assume sphere with radius 1!
            noise = np.sqrt(sig2) * np.random.randn(len(theta),1) * x
        else: noise = np.zeros(*x.shape)
        return noise  

    def distance_from_manifold(self, x):
        raise NotImplementedError

    def _draw_z(self, n):
        if self._latent_distribution == 'uniform':
            phi = np.random.random(n) * 2 * np.pi
            r = np.random.random(n) * 1
            
        return phi, r
    
    
    def _transform_z_to_disk(self, phi, r, mode='train'):
        c, a = 0, 1
        d1x = r *  np.cos(phi)
        d1y = r *  np.sin(phi)
        x = np.stack([ d1x + 2, d1y + 2], axis=1)  #shift disk
        params = np.ones(x.shape[0])
        if mode=='train':
            x = NumpyDataset(x, params)
        return x
    
    def _transform_z_to_stick(self, theta,phi, mode='train'):
        c, a = 0, 1
        d1x = r *  np.cos(phi)
        d1y = r *  np.sin(phi)
        x = np.stack([ d1x, d1y], axis=1) 
        params = np.ones(x.shape[0])
        if mode=='train':
            x = NumpyDataset(x, params)
        return x
    
    def _transform_z_to_x_nondeg(self, theta,phi, mode='train'):
        c, a = 0, 1
        d1x = -1*(c + a*np.sin(theta)) * np.cos(phi)
        d1y = -1*(c + a*np.sin(theta)) * np.sin(phi)
        d1z = (a * np.cos(theta))
        x = np.stack([ d1x, d1z,  d1y], axis=1) 
        params = np.ones(x.shape[0])
        if mode=='train':
            x = NumpyDataset(x, params)
        return x

    def _transform_x_to_z(self, x):
        raise NotImplementedError

    def _density(self, data):
        theta = data[0]
        phi = data[1]
        if self._latent_distribution == 'mixture':
            probs = 1/4* (2*np.exp(self.kappa*np.cos(2* (theta-self.mu31))) * np.exp(self.kappa*np.cos(phi-self.mu32)) *(1/(2*np.pi*i0(self.kappa))**2)
                 +2*np.exp(self.kappa*np.cos(2* (theta-self.mu11))) * np.exp(self.kappa*np.cos(phi-self.mu12)) *(1/(2*np.pi*i0(self.kappa))**2)   
                 +2*np.exp(self.kappa*np.cos(2* (theta-self.mu21))) * np.exp(self.kappa*np.cos(phi-self.mu22)) *(1/(2*np.pi*i0(self.kappa))**2)  
                 +2*np.exp(self.kappa*np.cos(2* (theta-self.mu41))) * np.exp(self.kappa*np.cos(phi-self.mu42)) *(1/(2*np.pi*i0(self.kappa))**2)
                 )
        elif self._latent_distribution == 'bigcheckerboard':
            s = np.pi/2-.2 # long side length
            def in_board(theta,phi,s):
                # z is lonlat
                lon = phi
                lat = theta
                if np.pi <= lon < np.pi+s or np.pi-2*s <= lon < np.pi-s:
                    return np.pi/2 <= lat < np.pi/2+s/2 or np.pi/2-s <= lat < np.pi/2-s/2
                elif np.pi-2*s <= lon < np.pi+2*s:
                    return np.pi/2+s/2 <= lat < np.pi/2+s or np.pi/2-s/2 <= lat < np.pi/2
                else:
                    return 0

            probs = np.zeros(theta.shape[0])
            for i in range(theta.shape[0]):
                probs[i] = in_board(theta[i],phi[i], s)

            probs /= np.sum(probs)   
        
        elif self._latent_distribution == 'unimodal':
            kappa, mu1, mu2  = 6, np.pi/2, np.pi
            probs = 2*np.exp(kappa*np.cos(2* (theta-mu1))) * np.exp(kappa*np.cos(phi-mu2)) *(1/(2*np.pi*i0(kappa))**2) 
        
        elif self._latent_distribution == 'correlated':
            kappa = 6.0
            mu11, kappa11 = 0, kappa        
            mu12, kappa12 = np.pi/2 , kappa             #northpole
            mu21, kappa21 = np.pi , kappa 
            mu22, kappa22 = 3*np.pi/2  , kappa          #southpole
            mu31, kappa31 = np.pi/2 , kappa
            mu32, kappa32 = 0 , kappa
            mu41, kappa41 = np.pi/2,  50
            mu42, kappa42 = np.pi , kappa
            
            prob = (1/(2*np.pi*i0(kappa41))) * 2*np.exp(kappa41*np.cos(2*(theta-mu41))) / (2*np.pi)
            probs = (1/3) * (prob 
                         +2*np.exp(kappa11*np.cos(2* (theta-mu11))) * np.exp(kappa12*np.cos(phi-mu12)) *(1/(2*np.pi*i0(kappa))**2)   
                         +2*np.exp(kappa21*np.cos(2* (theta-mu21))) * np.exp(kappa22*np.cos(phi-mu22)) *(1/(2*np.pi*i0(kappa))**2)   
                         )
                
        return probs
    
    def generate_grid(self,n,mode='sphere'):
        theta = np.linspace(0, np.pi, n)
        phi = np.linspace(0, 2*np.pi, n)
        # theta = np.linspace(1, 2, n)
        # phi = np.linspace(1, 2,n)
        latent = [theta,phi]
        xx, yy = np.meshgrid(theta, phi)
        
        grid = np.stack((xx.flatten(), yy.flatten()), axis=1)
        true_probs = self._density([grid[:,0],grid[:,1]])
        if mode == 'data_space':            
            data = self._transform_z_to_x(grid[:,0],grid[:,1],mode='test')
        else: data = [xx, yy]
        jacobians = np.sin(xx)
        # import pdb
        # pdb.set_trace()
        # jacobians[0,:]=1
        # jacobians[n-1,:]=1
        return data, latent, true_probs, jacobians, 1
        

    def calculate_sigma_bound(self,theta, phi): #theta=v, phi = u
        def circle_term_theta(z,alpha):
            return self.kappa * 4* (self.kappa*np.sin(2*(z-alpha))**2 - np.cos(2*(z-alpha)) )
       
        def circle_term_phi(z,alpha):
            return self.kappa * (self.kappa*np.sin(z-alpha)**2 - np.cos(z-alpha) )
        
        def jacobian(theta):                   
            return np.sin(theta)
        
        def unimodal(theta,phi,mu,m):
            return 2*np.exp(self.kappa*np.cos(2*(theta-mu))+self.kappa*np.cos(phi-m)) * (1/(2*np.pi*i0(self.kappa))**2) 
       
        if self._latent_distribution == 'mixture':
            bound = 2 * self._density([theta,phi]) / ( unimodal(theta,phi,self.mu11,self.mu12) * (circle_term_theta(theta,self.mu11)/jacobian(theta)**2 + circle_term_phi(phi,self.mu12) )  
                                                      +unimodal(theta,phi,self.mu11,self.mu21) * (circle_term_theta(theta,self.mu21)/jacobian(theta)**2 + circle_term_phi(phi,self.mu22) )
                                                      +unimodal(theta,phi,self.mu31,self.mu32) * (circle_term_theta(theta,self.mu31)/jacobian(theta)**2 + circle_term_phi(phi,self.mu32) ) )
        
        elif self._latent_distribution == 'correlated':
            bound = 1000 # (2 * self._a**2 * jacobian(theta)**2 ) / ( circle_term(phi,theta,self._mu) * (self._a**2 + jacobian(theta)**2)  )
       
        return bound

    def calculate_gauss_curvature(self,theta,phi):
        return np.ones(*theta.shape)