# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 11:16:30 2021
BNAF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math
# --------------------
# Model components
# --------------------
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, data_dim):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.data_dim = data_dim
        

        # Notation:
        # BNAF weight calculation for (eq 8): W = g(W) * M_d + W * M_o
        #   where W is block lower triangular so model is autoregressive,
        #         g = exp function; M_d is block diagonal mask; M_o is block off-diagonal mask.
        # Weight Normalization (Salimans & Kingma, eq 2): w = g * v / ||v||
        #   where g is scalar, v is k-dim vector, ||v|| is Euclidean norm
        # ------
        # Here: pre-weight norm matrix is v; then: v = exp(weight) * mask_d + weight * mask_o
        #       weight-norm scalar is g: out_features dimensional vector (here logg is used instead to avoid taking logs in the logdet calc.
        #       then weight-normed weight matrix is w = g * v / ||v||
        #
        #       log det jacobian of block lower triangular is taking block diagonal mask of
        #           log(g*v/||v||) = log(g) + log(v) - log(||v||)
        #                          = log(g) + weight - log(||v||) since v = exp(weight) * mask_d + weight * mask_o

        weight = torch.zeros(out_features, in_features)
        mask_d = torch.zeros_like(weight)
        mask_o = torch.zeros_like(weight)
        for i in range(data_dim):
            # select block slices
            h     = slice(i * out_features // data_dim, (i+1) * out_features // data_dim)
            w     = slice(i * in_features // data_dim,  (i+1) * in_features // data_dim)
            w_row = slice(0,                            (i+1) * in_features // data_dim)
            # initialize block-lower-triangular weight and construct block diagonal mask_d and lower triangular mask_o
            nn.init.kaiming_uniform_(weight[h,w_row], a=math.sqrt(5))  # default nn.Linear weight init only block-wise
            mask_d[h,w] = 1
            mask_o[h,w_row] = 1

        mask_o = mask_o - mask_d  # remove diagonal so mask_o is lower triangular 1-off the diagonal

        self.weight = nn.Parameter(weight)                          # pre-mask, pre-weight-norm
        self.logg = nn.Parameter(torch.rand(out_features, 1).log()) # weight-norm parameter
        self.bias = nn.Parameter(nn.init.uniform_(torch.rand(out_features), -1/math.sqrt(in_features), 1/math.sqrt(in_features)))  # default nn.Linear bias init
        self.register_buffer('mask_d', mask_d)
        self.register_buffer('mask_o', mask_o)
        
        #W_T_W = torch.rand(data_dim, device='cpu')         #1 to be replacd by manifold dim
        #self.W_T_W = nn.Parameter(W_T_W)

    def forward(self, x, sum_logdets):
        # 1. compute BNAF masked weight eq 8
        v = self.weight.exp() * self.mask_d + self.weight * self.mask_o
        # 2. weight normalization
        v_norm = v.norm(p=2, dim=1, keepdim=True)
        w = self.logg.exp() * v / v_norm
        # 3. compute output and logdet of the layer
        out = F.linear(x, w, self.bias)
        logdet = self.logg + self.weight - 0.5 * v_norm.pow(2).log()
        logdet = logdet[self.mask_d.bool()]
        logdet = logdet.view(1, self.data_dim, out.shape[1]//self.data_dim, x.shape[1]//self.data_dim) \
                       .expand(x.shape[0],-1,-1,-1)  # output (B, data_dim, out_dim // data_dim, in_dim // data_dim)

        # 4. sum with sum_logdets from layers before (BNAF section 3.3)
        # Compute log det jacobian of the flow (eq 9, 10, 11) using log-matrix multiplication of the different layers.
        # Specifically for two successive MaskedLinear layers A -> B with logdets A and B of shapes
        #  logdet A is (B, data_dim, outA_dim, inA_dim)
        #  logdet B is (B, data_dim, outB_dim, inB_dim) where outA_dim = inB_dim
        #
        #  Note -- in the first layer, inA_dim = in_features//data_dim = 1 since in_features == data_dim.
        #            thus logdet A is (B, data_dim, outA_dim, 1)
        #
        #  Then:
        #  logsumexp(A.transpose(2,3) + B) = logsumexp( (B, data_dim, 1, outA_dim) + (B, data_dim, outB_dim, inB_dim) , dim=-1)
        #                                  = logsumexp( (B, data_dim, 1, outA_dim) + (B, data_dim, outB_dim, outA_dim), dim=-1)
        #                                  = logsumexp( (B, data_dim, outB_dim, outA_dim), dim=-1) where dim2 of tensor1 is broadcasted
        #                                  = (B, data_dim, outB_dim, 1)

        sum_logdets = torch.logsumexp(sum_logdets.transpose(2,3) + logdet, dim=-1, keepdim=True)

        return out, sum_logdets


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, sum_logdets):
        # derivation of logdet:
        # d/dx tanh = 1 / cosh^2; cosh = (1 + exp(-2x)) / (2*exp(-x))
        # log d/dx tanh = - 2 * log cosh = -2 * (x - log 2 + log(1 + exp(-2x)))
        logdet = -2 * (x - math.log(2) + F.softplus(-2*x))
        sum_logdets = sum_logdets + logdet.view_as(sum_logdets)
        return x.tanh(), sum_logdets

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x):
        sum_logdets = torch.zeros(1, x.shape[1], 1, 1, device=x.device)
        for module in self:
            x, sum_logdets = module(x, sum_logdets)
        return x, sum_logdets.squeeze()


# --------------------
# Model
# --------------------

class BlockNeuralAutoregressiveFlow(nn.Module):
    def __init__(self, data_dim, n_hidden, hidden_dim, use_batch_norm=False, uniform_target=False):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(data_dim))
        self.register_buffer('base_dist_var', torch.ones(data_dim))
        self.uniform_target = uniform_target

        # construct model
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(data_dim, eps=1e-3)
        else: self.batch_norm = False
        
        modules = []
        modules += [MaskedLinear(data_dim, hidden_dim, data_dim), Tanh()]
        for _ in range(n_hidden):
            modules += [MaskedLinear(hidden_dim, hidden_dim, data_dim), Tanh()]
        if uniform_target:
            modules += [MaskedLinear(hidden_dim, data_dim, data_dim), Tanh()]
        else:   
            modules += [MaskedLinear(hidden_dim, data_dim, data_dim)]
        
        self.net = FlowSequential(*modules)
 
        # modules2 = []
        # modules2 += [MaskedLinear(data_dim, hidden_dim, data_dim), Tanh()]
        # for _ in range(n_hidden):
        #     modules2 += [MaskedLinear(hidden_dim, hidden_dim, data_dim), Tanh()]
        # modules2 += [MaskedLinear(hidden_dim, data_dim, data_dim)]
        # self.net2 = FlowSequential(*modules)
        
        # self.perm = torch.tensor([2, 1, 0])

    @property
    def base_dist(self):
        if self.uniform_target:
            dist =  D.uniform.Uniform(-1,1)
        else:
            dist = D.Normal(self.base_dist_mean, self.base_dist_var)
        return dist
    
    def forward(self, x):
        if self.batch_norm:
            outputs = self.batch_norm(x)
        else:
            outputs = x
        # out, logdets = self.net(outputs)
        # out_ = out[:,self.perm]
        # out2, logdets2 = self.net2(out_)
        
        return self.net(outputs)  #out2, logdets+logdets2 #s
    
    def encode(self, x):
        if self.batch_norm:
            outputs = self.batch_norm(x)
        else:
            outputs = x
        # out, logdets = self.net(outputs)
        # out_ = out[:,self.perm]
        # out2, logdets2 = self.net2(out_)
        z, _ = self.net(outputs)
        return z 

class BlockNeuralAutoregressiveFlow_uniform(nn.Module):
    def __init__(self, data_dim, n_hidden, hidden_dim, use_batch_norm=False):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(data_dim))
        self.register_buffer('base_dist_var', torch.ones(data_dim))

        # construct model
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(data_dim, eps=1e-3)
        else: self.batch_norm = False
        
        modules = []
        modules += [MaskedLinear(data_dim, hidden_dim, data_dim), Tanh()]
        for _ in range(n_hidden):
            modules += [MaskedLinear(hidden_dim, hidden_dim, data_dim), Tanh()]
        modules += [MaskedLinear(hidden_dim, data_dim, data_dim), Tanh()]
        self.net = FlowSequential(*modules)
 
        # modules2 = []
        # modules2 += [MaskedLinear(data_dim, hidden_dim, data_dim), Tanh()]
        # for _ in range(n_hidden):
        #     modules2 += [MaskedLinear(hidden_dim, hidden_dim, data_dim), Tanh()]
        # modules2 += [MaskedLinear(hidden_dim, data_dim, data_dim)]
        # self.net2 = FlowSequential(*modules)
        
        # self.perm = torch.tensor([2, 1, 0])

    @property
    def base_dist(self):
        return D.uniform.Uniform(-1,1)

    def forward(self, x):
        if self.batch_norm:
            outputs = self.batch_norm(x)
        else:
            outputs = x
        # out, logdets = self.net(outputs)
        # out_ = out[:,self.perm]
        # out2, logdets2 = self.net2(out_)
        
        return self.net(outputs)  #out2, logdets+logdets2 #s
