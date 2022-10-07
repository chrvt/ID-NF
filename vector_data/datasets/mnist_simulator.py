#! /usr/bin/env python

import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator
from .utils import NumpyDataset
from scipy.special import i0
import torch

logger = logging.getLogger(__name__)


class SphereSimulator(BaseSimulator):
    def __init__(self, latent_dim=2, data_dim=784, kappa=6.0, epsilon=0.,latent_distribution='mix_of_vonMises',noise_type='gaussian'):
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
        return 'MNIST'
    
    def dataset(self):
        return 'MNIST'+'_'+self._latente_distribution

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

    def sample_and_noise(self,n,parameters=None, sig2 = 0.0, mode='numpy'):
        theta, phi = self._draw_z(n)
        x = self._transform_z_to_x(theta,phi,mode='numpy')
        noise = self.create_noise(x,theta,phi,sig2)
        return np.stack([x, noise],axis=-1)   

    def create_noise(self,x,theta,phi,sig2):
        if self._noise_type == 'gaussian':
            noise = np.sqrt(sig2) * np.random.randn(*x.shape)
        elif self._noise_type == 'normal': #assume sphere with radius 1!
            noise = np.sqrt(sig2) * np.random.randn(len(theta),1) * x
        else: noise = np.zeros(*x.shape)
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
            
            theta1 = np.random.vonmises((self.mu11-np.pi/2)*2,self.kappa,n_samples_1 )/2 + np.pi/2
            phi1 = np.random.vonmises(self.mu12-np.pi,self.kappa,n_samples_1 ) + np.pi
            theta2 = np.random.vonmises((self.mu21-np.pi/2)*2,self.kappa,n_samples_2 )/2 + np.pi/2
            phi2 = np.random.vonmises(self.mu22-np.pi,self.kappa,n_samples_2 ) + np.pi
            theta3 = np.random.vonmises((self.mu31-np.pi/2)*2,self.kappa,n_samples_3 )/2 + np.pi/2
            phi3 = np.random.vonmises(self.mu32-np.pi,self.kappa,n_samples_3 ) + np.pi
            theta4 = np.random.vonmises((self.mu41-np.pi/2)*2,self.kappa,n_samples_4 )/2 + np.pi/2
            phi4 = np.random.vonmises(self.mu42-np.pi,self.kappa,n_samples_4 ) + np.pi
            
            theta = np.concatenate([theta1,theta2,theta3,theta4],axis=0)
            phi = np.concatenate([phi1,phi2,phi3,phi4],axis=0)     
        # elif self._latent_distribution == 'mix_of_vonMises':
        elif self._latent_distribution == 'bigcheckerboard':
            # s = np.pi/2-.2 # long side length
            # offsets = [(0,0), (s, s/2), (s, -s/2), (0, -s), (-s, s/2), (-s, -s/2), (-2*s, 0), (-2*s, -s)]
            # # offsets = torch.tensor([o for o in offsets])            
            
            # x1 = np.random.rand(n) * s + np.pi
            # x2 = np.random.rand(n) * s/2 + np.pi/2
            
            # samples = np.stack([x1, x2], axis=1)
            # off = offsets[np.random.randint(len(offsets), size=n)]

            # samples += off
            
            s = np.pi/2-.2 # long side length
            offsets = [(0,0), (s, s/2), (s, -s/2), (0, -s), (-s, s/2), (-s, -s/2), (-2*s, 0), (-2*s, -s)]
            offsets = torch.tensor([o for o in offsets])
    
            # (x,y) ~ uniform([pi,pi + s] times [pi/2, pi/2 + s/2])
            x1 = torch.rand(n) * s + np.pi
            x2 = torch.rand(n) * s/2 + np.pi/2
    
            samples = torch.stack([x1, x2], dim=1)
            off = offsets[torch.randint(len(offsets), size=(n,))]
    
            samples += off
            
            phi = samples[:,0].cpu().numpy()
            theta = samples[:,1].cpu().numpy()       
        elif self._latent_distribution == 'unimodal':
            kappa, mu1, mu2  = 6, np.pi/2, np.pi
            theta = (np.random.vonmises(0,kappa,n)+np.pi)/2 #(2*np.pi)
            phi   = (np.random.vonmises(0,kappa,n)+np.pi)
           
        elif self._latent_distribution == 'correlated':
            kappa, kappa1 = 6.0, 50
            mu11, mu12, mu21, mu22, mu31, mu32 = 0, 0, 0, np.pi/2, np.pi, 3*np.pi/2
            
            n_samples = np.random.multinomial(n,[1/3]*3,size=1)
            n_samples_1 = n_samples[0,0]
            n_samples_2 = n_samples[0,1]
            n_samples_3 = n_samples[0,2]
            
            phi1 = (np.random.vonmises(mu11,0,n_samples_1)+np.pi)
            theta1 = (np.random.vonmises(mu12,kappa1,n_samples_1)+np.pi)/2
            
            theta2 = np.random.vonmises((mu11-np.pi/2)*2,kappa,n_samples_2 )/2 + np.pi/2
            phi2 = np.random.vonmises(mu22-np.pi,kappa,n_samples_2 ) + np.pi
            theta3 = np.random.vonmises((mu31-np.pi/2)*2,kappa,n_samples_3 )/2 + np.pi/2
            phi3 = np.random.vonmises(mu32-np.pi,kappa,n_samples_3 ) + np.pi
            
            theta = np.concatenate([theta1,theta2,theta3],axis=0)
            phi = np.concatenate([phi1,phi2,phi3],axis=0)      
           
        return theta, phi
    
    
    def _transform_z_to_x(self, theta,phi, mode='train'):
        c, a = 0, 1
        d1x = (c + a*np.sin(theta)) * np.cos(phi)
        d1y = (c + a*np.sin(theta)) * np.sin(phi)
        d1z = (a * np.cos(theta))
        x = np.stack([ d1x, d1y, d1z], axis=1) 
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
            bound = 0 # (2 * self._a**2 * jacobian(theta)**2 ) / ( circle_term(phi,theta,self._mu) * (self._a**2 + jacobian(theta)**2)  )
       
        return bound
