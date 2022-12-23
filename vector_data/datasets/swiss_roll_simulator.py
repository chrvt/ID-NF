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
        u, v = self._draw_z(n)
        x = self._transform_z_to_x(u,v,mode='numpy')
        # z_hat =  self._transform_x_to_z(x)
        # import pdb
        # pdb.set_trace()
        return x

    def sample_ood(self, n, parameters=None):
        x = self.sample(n)
        noise = np.sqrt(self._epsilon) * np.random.normal(size=(n, 2))
        return x + noise
    
    def sample_and_noise(self,n,sig2 = 0.0, mode='numpy'):
        # import pdb
        # pdb.set_trace()
        u, v = self._draw_z(n)
        x = self._transform_z_to_x(u,v,mode='numpy')
        # import pdb
        # pdb.set_trace()
        noise = self.create_noise(np.array(x,copy=True),u,v,sig2)
        # import torch
        # nois_torch = torch.from_numpy(noise).float()
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(10,10))
        # ax1 = fig.add_subplot(111)
        # ax1.scatter(u, v,s=1)
        return np.stack([x, noise],axis=-1)   #create new dataset for loader

    def create_noise(self,x,u,v,sig2):
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
        # import pdb
        # pdb.set_trace() 
        return data

    def _transform_x_to_normal(self,data):
        data[:,1] = 0  #set y to zero to have vanilla spiral
        norm = np.linalg.norm(data,axis=1).reshape([data.shape[0],1])
        z = 3 * norm
        e_r = data / norm
        R = np.array(([0,0,-1],[0,0,0],[+1,0,0])) 
        e_phi = +1*np.matmul(e_r,R)
        x_norm = (e_r + z * e_phi)/3 
        normal = np.matmul(x_norm,R)
        return normal

    # def _transform_z_to_spiral(self,u,v):
    #     t = 3/2 * np.pi * (1+2*v)
    #     x = np.sin(t) + t*np.cos(t)
    #     y = np.zeros(*v.shape)
    #     z = -np.cos(t) + t*np.sin(t)
    #     x = np.stack([x,y,z], axis=1) 
    #     return x / np.linalg.norm(x)
    
    def _translate(self,x): #from [0,1] to [-pi,pi]
        return 2*np.pi*x-np.pi

    def _translate_inverse(self,u): #from [-pi,pi] to [0,1]  
        return (u+np.pi)/(2*np.pi)

    def _transform_x_to_z(self, x):
        bs = x.shape[0]
        z1 = (x[:,1]/21).reshape([bs,1]) 
        z2 = ((np.linalg.norm(x[:,[0,2]],axis=1)-1.5*np.pi)/(3*np.pi)).reshape([bs,1]) 
        z = np.concatenate([z1,z2],axis=1)
        return z

    def _density(self, data):
        u_ = data[0]
        v_ = data[1]
        if self._latent_distribution == 'mixture':
            kappa, mu11, mu12, mu21, mu22, mu31, mu32 = 2, 0.1, 0.1, 0.5, 0.8, 0.8, 0.8 
            probs= (1/3)*(np.exp(kappa*np.cos( self._translate(u_) - self._translate(mu11) ) +kappa*np.cos( self._translate(v_) - self._translate(mu12) ))
                        + np.exp(kappa*np.cos( self._translate(u_) - self._translate(mu21) ) +kappa*np.cos( self._translate(v_) - self._translate(mu22) ))
                        + np.exp(kappa*np.cos( self._translate(u_) - self._translate(mu31) ) +kappa*np.cos( self._translate(v_) - self._translate(mu32) ))) / i0(kappa)**2  #*(1/(2*np.pi*i0(kappa))**2) * (2*np.pi)**2 
        elif self._latent_distribution == 'correlated':
            kappa, mu = 5, 0.0
            probs = 2*np.pi * np.exp((kappa)*np.cos( self._translate(v_) - self._translate(u_) )) *(1/(2*np.pi*i0(kappa))) 
        elif self._latent_distribution == 'unimodal':
            mu1, mu2, kappa = 0.5, 0.8, 2 
            probs = (2*np.pi)**2 * np.exp(kappa*np.cos( self._translate(u_) - self._translate(mu1) ) + kappa*np.cos(self._translate(v_) - self._translate(mu2)) ) *(1/(2*np.pi*i0(kappa))**2) 
        return probs
    
    def generate_grid(self,n,mode='data_space'):
        multiplier = 1
        u_ = np.linspace(0, 1, n)
        v_ = np.linspace(0, 1, n*multiplier)
        latent = [u_,v_]
        xx, yy = np.meshgrid(u_, v_)
        
        grid = np.stack((xx.flatten(), yy.flatten()), axis=1)
        true_probs = self._density([grid[:,0],grid[:,1]])
        if mode == 'data_space':            
            data = self._transform_z_to_x(grid[:,0],grid[:,1],mode='test')
        else: data = [xx, yy]
        t = 3/2*np.pi*(1+2*yy)
        jacobians = 3*np.pi*21*np.sqrt(1+t**2)
        # import pdb
        # pdb.set_trace()
        return data, latent, true_probs, jacobians, multiplier #torch.tensor(data)
        

    def calculate_sigma_bound(self,u_, v_): #theta=v, phi = u
        def circle_term(u,v,alpha=0,sign=-1,kappa=2):
            return (2*np.pi)**2 * kappa * (kappa*np.sin( 2*np.pi* (v-u- alpha))**2 + sign * np.cos( 2*np.pi*(v-u-alpha)))
        def jacobian(theta):                   
            return self._c+self._a*np.cos(theta)
        def unimodal(u,v,mu,m):
            kappa = 2
            return (1/3) * np.exp(kappa*np.cos( self._translate(u) - self._translate(mu) ) +kappa*np.cos( self._translate(v) - self._translate(m) )) / i0(kappa)**2
       
        if self._latent_distribution == 'mixture':
            kappa, mu11, mu12, mu21, mu22, mu31, mu32 = 2, 0.1, 0.1, 0.5, 0.8, 0.8, 0.8 
            t = 3/2*np.pi*(1+2*v_)
            bound =(2 * self._density([u_,v_]))/ ( unimodal(u_,v_,mu11,mu12) * ( circle_term(-u_,0,alpha=mu11,kappa=kappa) / 21**2 + circle_term(0,v_,alpha=mu12,kappa=kappa) / ( (3*np.pi)**2 * (1+t**2))  )   
                                                  +unimodal(u_,v_,mu11,mu12) * ( circle_term(-u_,0,alpha=mu21,kappa=kappa) / 21**2 + circle_term(0,v_,alpha=mu22,kappa=kappa) / ( (3*np.pi)**2 * (1+t**2))  )  
                                                  +unimodal(u_,v_,mu11,mu12) * ( circle_term(-u_,0,alpha=mu31,kappa=kappa) / 21**2 + circle_term(0,v_,alpha=mu32,kappa=kappa) / ( (3*np.pi)**2 * (1+t**2))  )  
                                                 )
        
        elif self._latent_distribution == 'correlated':
            t = 3/2*np.pi*(1+2*v_)
            bound_ = 0.5 * ( circle_term(u_,v_,kappa=5) / 21**2 + circle_term(u_,v_,kappa=5) / ( (3*np.pi)**2 * (1+t**2))  )
            bound = 1/bound_
       
        return bound
    
    def calculate_gauss_curvature(self, u,v):
        a = 1.5*np.pi
        E = 3*np.pi*np.sqrt(1+(a+3*np.pi*v)**2)   
        F = 0
        G = 21
        e = 0
        f = 0 
        # #----
        #Wolfram alpha input: {3*pi*cos\(40)a+3*pi*v\(41)-3*pi*\(40)a+3*pi*v\(41)*sin\(40)a+3*pi*v\(41),0,3*pi*sin\(40)a+3*pi*v\(41)+3*pi*\(40)a+3*pi*v\(41)*cos\(40)a+3*pi*v\(41)}x{0,21\(44)0}
        # x_vv = np.array([-(3*np.pi)**2 * np.sin(a+3*np.pi*v) - 3*np.pi * (3*np.pi * np.sin(a+3*np.pi*v) + (a+3*np.pi*v) * np.cos(a+3*np.pi*v)),np.zeros(len(v)), (3*np.pi)**2 * np.cos(a+3*np.pi*v) + 3*np.pi * (3*np.pi * np.cos(a+3*np.pi*v) - (a+3*np.pi * v) * np.sin(a+3*np.pi*v))])
        # x_u_wedge_x_v = np.array([-63*a*np.pi*np.cos(a+3*np.pi*v)-189*np.pi**2 * v* np.cos(a+3*np.pi*v) - 63*np.pi*np.sin(a+3*np.pi*v),np.zeros(len(v)), 63*a*np.pi*np.cos(a+3*np.pi*v)-189*np.pi**2 * v* np.sin(a+3*np.pi*v) - 63*np.pi*np.sin(a+3*np.pi*v)] )

        # x_norm = x_u_wedge_x_v / np.linalg.norm(x_u_wedge_x_v,axis=0)
        
        # g = np.zeros(len(v))
        # for k in range(len(v)):
        #     g[k] = np.dot(x_norm[:,k],x_vv[:,k])
            
        return np.zeros(len(v)) #(e*g-f**2) / (E*G-F**2)