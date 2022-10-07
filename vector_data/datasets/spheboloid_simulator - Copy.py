#! /usr/bin/env python
# torus: https://en.wikipedia.org/wiki/Torus

import numpy as np
from scipy.stats import norm
import logging
from .base import BaseSimulator
from .utils import NumpyDataset
# from .utils import _cartesian_to_spherical
from scipy.special import i0
from scipy.stats import expon

logger = logging.getLogger(__name__)


class SpheRoidSimulator(BaseSimulator):
    def __init__(self, latent_dim=2, data_dim=3, scale=0.4, kappa=10, epsilon=0., latent_distribution='correlated', noise_type=None):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._epsilon = epsilon
        self._scale = scale
        self._kappa = kappa
        
        self._latent_distribution = latent_distribution
        self._noise_type = noise_type
        

            
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
        z1_, z2_ = self._draw_z(n)
        x = self._transform_z_to_x(z1_,z2_,mode='numpy')
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = self._epsilon * np.random.normal(size=(n, 2))
        return x + noise

    def sample_and_noise(self,n,parameters=None, sig2 = 0.0, mode='numpy'):
        z1_, z2_  = self._draw_z(n)
        x = self._transform_z_to_x(z1_,z2_ ,mode='numpy')
        
        noise = self.create_noise(x,z1_,z2_,sig2)
        
        return np.stack([x, noise],axis=-1)   
    
    def create_noise(self,x,z1,z2,sig2):
        if self._noise_type == 'gaussian':
            noise = np.sqrt(sig2) * np.random.randn(*x.shape)
        elif self._noise_type == 'normal':
            idx_hyp = np.where(z1 < 0 )[0]
            idx_sph = np.where(z1 >=0 )[0]  
            noise = np.zeros([len(z1),self._data_dim])
            noise[idx_sph,:] = np.sqrt(sig2) * np.random.randn(len(idx_sph),1) * x[idx_sph,:]   #self._add_normal_noise_sphere(x)
            #----------
            normal_ = self._transform_z_to_hyperboloid(z1[idx_hyp],z2[idx_hyp],sign=-1)
            noise[idx_hyp,:] = np.sqrt(sig2) * np.random.randn(len(idx_hyp),1) * (normal_ / np.linalg.norm(normal_ ,axis=1).reshape(normal_.shape[0],1) )
        else: noise = np.zeros([len(z1),self._data_dim])
        return noise         
            
    def distance_from_manifold(self, x):
        raise NotImplementedError

    def _draw_z_sphere(self, n_smpl):
        if self._latent_distribution == 'correlated':
            theta_smpl = np.random.exponential(scale=self._scale, size=n_smpl)
            phi_smpl = np.random.vonmises(-theta_smpl,self._kappa,n_smpl) + np.pi
            
        return np.stack([theta_smpl,phi_smpl],axis=1)
 
    def _draw_z_hyp(self, n_smpl):
        if self._latent_distribution == 'correlated':
            v_smpl = -1*np.random.exponential(scale=self._scale, size=n_smpl)
            psi_smpl = np.random.vonmises(np.abs(v_smpl),10,n_smpl) +  np.pi

        return np.stack([v_smpl,psi_smpl],axis=1)           
        

    def _draw_z(self, n):
        if self._latent_distribution == 'correlated':
            n_samples = np.random.multinomial(n,[1/2]*2,size=1)
            n_sph = n_samples[0,0]
            n_hyp = n_samples[0,1]
            
            z_sph = self._draw_z_sphere(n_sph)
            z_hyp = self._draw_z_hyp(n_hyp)
            
            z1 = np.concatenate([z_sph[:,0],z_hyp[:,0]])
            z2 = np.concatenate([z_sph[:,1],z_hyp[:,1]])
            
        elif self._latent_distribution == 'unimodal':
            raise NotImplementedError
        
        return z1, z2

    def _transform_z_to_hyperboloid(self, v,psi,sign=1):
        a,b,c = 1, 1, 1 
        x = -a*(np.cosh(np.abs(v)))*np.cos(psi)
        y = -b*(np.cosh(np.abs(v)))*np.sin(psi)
        z = sign * c*np.sinh(np.abs(v))
        samples = np.stack([x,y,z], axis=1) 
        return samples

    def _transform_z_to_sphere(self, theta,phi):
        c, a = 0, 1
        x = (c + a*np.cos(theta+np.pi)) * np.cos(phi)
        y = (c + a*np.cos(theta+np.pi)) * np.sin(phi)
        z = a * np.sin(theta+np.pi)
        x = np.stack([x,y,z], axis=1) 
        return x     
    
    def _transform_z_to_x(self, z1, z2, sign=1, mode='numpy'):
        idx_hyp = np.where(z1 < 0 )[0]
        idx_sph = np.where(z1 >=0 )[0]       
        
        x = np.zeros([len(z1),3])
        
        x[idx_sph,:] = self._transform_z_to_sphere(z1[idx_sph],z2[idx_sph])
        x[idx_hyp,:] = self._transform_z_to_hyperboloid(z1[idx_hyp],z2[idx_hyp])
 
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

    def _density_sphere(self, theta,phi):
        if self._latent_distribution == 'correlated':
            probs_phi =  np.exp((self._kappa)*np.cos(phi- np.pi + theta)) *(1/(2*np.pi*i0(self._kappa))) #10/(2*np.pi)#n
            probs_theta = expon.pdf(theta,scale=self._scale) #np.ones(len(theta))*2/np.pi  #
            probs = probs_theta * probs_phi 
                         
        return probs
    
    def _density_hyperboloid(self, v,psi):
        if self._latent_distribution == 'correlated':
            probs_theta = 1*( np.exp(self._kappa*np.cos(psi-np.pi-np.abs(v)))) *(1/(2*np.pi*i0(self._kappa)))
            probs_v = expon.pdf(np.abs(v),scale=self._scale) # np.ones(len(v))*2/np.pi 
            probs = probs_v * probs_theta
            
        return probs
    
    def _density(self, z1, z2):
        idx_hyp = np.where(z1 < 0 )[0]
        idx_sph = np.where(z1 >=0 )[0] 
        probs = np.zeros(len(z1))
        if self._latent_distribution == 'correlated':
            probs[idx_sph] = 0.5 * self._density_sphere(z1[idx_sph],z2[idx_sph])
            probs[idx_hyp] = 0.5 * self._density_hyperboloid(z1[idx_hyp],z2[idx_hyp])
                        
        elif self._latent_distribution == 'unimodal':
            raise NotImplementedError          
            
        return probs
    
    def generate_grid(self,n,mode='data_space'):
        z1_a = np.linspace(-2, 0, int(n/2)) #[:-1]  
        z1_b = np.linspace(0, 2, int(n/2)+1)[1:]  #np.pi/2
        z1_  = np.concatenate([z1_a,z1_b])
        # import pdb
        # pdb.set_trace()
        z2_ = np.linspace(0, 2*np.pi, n)
        
        latent = [z1_, z2_]
        xx, yy = np.meshgrid(z1_, z2_)
        
        grid = np.stack((xx.flatten(), yy.flatten()), axis=1)
        true_probs = self._density(grid[:,0],grid[:,1])
        if mode == 'data_space':            
            data = self._transform_z_to_x(grid[:,0],grid[:,1],mode='test')
        else: data = [xx, yy]
        jacobians = np.zeros([n,n])
        idx_hyp = np.where(xx<0)
        idx_sph = np.where(xx>=0)
        # import pdb
        # pdb.set_trace()
        jacobians[idx_hyp[0],idx_hyp[1]] = np.sqrt( ( np.sinh(xx[idx_hyp[0],idx_hyp[1]])**2 +  np.cosh(xx[idx_hyp[0],idx_hyp[1]])**2 ) *  np.cosh(xx[idx_hyp[0],idx_hyp[1]])**2 )
        jacobians[idx_sph[0],idx_sph[1]] = np.cos(xx[idx_sph[0],idx_sph[1]])

        return data, latent, true_probs, jacobians, 1
        


