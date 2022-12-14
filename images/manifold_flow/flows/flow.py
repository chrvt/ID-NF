import logging

from manifold_flow.utils.various import product
from manifold_flow import distributions
from manifold_flow.flows import BaseFlow
import torch
import numpy as np
logger = logging.getLogger(__name__)


class Flow(BaseFlow):
    """ Ambient normalizing flow (AF) """

    def __init__(self, data_dim, transform):
        super(Flow, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = data_dim
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(self.latent_dim)

        self.latent_distribution = distributions.StandardNormal((self.total_latent_dim,))
        self.transform = transform

        self._report_model_parameters()

    def forward(self, x, context=None):
        """ Transforms data point to latent space, evaluates log likelihood """

        # Encode
        u, log_det = self._encode(x, context=context)

        # Decode
        x = self.decode(u, context=context)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det

        return x, log_prob, u

    def encode(self, x, context=None):
        """ Encodes data point to latent space """

        u, _ = self._encode(x, context=context)
        return u

    def decode(self, u, context=None):
        """ Encodes data point to latent space """

        x, _ = self.transform.inverse(u, context=context)
        return x

    def log_prob(self, x, context=None):
        """ Evaluates log likelihood """

        # Encode
        u, log_det = self._encode(x, context)

        # Log prob
        log_prob = self.latent_distribution._log_prob(u, context=None)
        log_prob = log_prob + log_det

        return log_prob

    def sample(self, u=None, n=1, context=None):
        """ Generates samples from model """

        if u is None:
            u = self.latent_distribution.sample(n, context=None)
        x = self.decode(u, context=context)
        return x

    def _encode(self, x, context=None):
        u, log_det = self.transform(x, context=context)
        return u, log_det
    
    def sample_inflation(self, u=None, n=1, sigma_max=1, cotext=None):
        """ Generates samples from model, increasingly away from original sample """
                
        if u is None:
            u_dir = sigma_max * self.latent_distribution.sample(1, context=None)
        
        u_zero = torch.zeros([1,self.total_latent_dim])
        
        u_t = torch.zeros([n,self.total_latent_dim])
        
        sigma = np.linspace(0,sigma_max, n)
        dt = torch.linspace(0,1,n)
        
        for k in range(n):
            u_t[k,:] = u_zero[0,:]*dt[k] + (1-dt[k])*u_dir[0,:]   
            
        return u_t