"""
Implementation of Inflation/Deflation method based on Block Neural Autoregressive Flow
http://arxiv.org/abs/1904.04676
"""
import warnings
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
import sys

import math
import numpy as np
import os
import time
import argparse
import pprint
from functools import partial
from scipy.special import gamma
#import matplotlib
#matplotlib.use('Agg')
import matplotlib
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from tqdm import tqdm
import pdb

from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.special import i0
from scipy import integrate

from datasets import load_simulator, SIMULATORS
from models import BlockNeuralAutoregressiveFlow as BNAF
from plotting import plt_latent_distribution as plot_latent

from utils import load_checkpoint

from torch.utils.data import DataLoader
# from utils import create_filename

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# general
parser.add_argument('--train', action='store_true', help='Train a flow.')
parser.add_argument('--plot', action='store_true', help='Plot a flow and target density.')
parser.add_argument('--calculate_KS', action='store_true', help='Caclulates KS_test at the end of the training.')
parser.add_argument('--restore_file', action='store_true', help='Restore model.')
parser.add_argument('--debug', action='store_true', help='Debug mode: for more infos')
#parser.add_argument('--output_dir', default='results\{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--output_dir', default='./results')  #.format(os.path.splitext(__file__)[0]))
parser.add_argument('--cuda', default=0, type=int, help='Which GPU to run on.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
# target density
parser.add_argument('--dataset', type=str, help='Which potential function to approximate.')
parser.add_argument("--latent_distribution", type=str, default=None, help="Latent distribution (for datasets where that is variable)")
# model parameters
parser.add_argument('--data_dim', type=int, default=3, help='Dimension of the data.')
parser.add_argument('--latent_dim', type=int, default=2, help='Dimension of manifold.')
parser.add_argument('--hidden_dim', type=int, default=210, help='Dimensions of hidden layers.')
parser.add_argument('--n_hidden', type=int, default=3, help='Number of hidden layers.')
# training parameters
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--n_gradient_steps', type=int, default=2000, help='Number of steps to train.')
parser.add_argument('--batch_size', type=int, default=200, help='Training batch size.')
parser.add_argument('--lr', type=float, default=1e-1, help='Initial learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay.')
parser.add_argument('--lr_patience', type=float, default=2000, help='Number of steps before decaying learning rate.')
parser.add_argument('--log_interval', type=int, default=50, help='How often to save model and samples.')
parser.add_argument("--noise_type", type=str, default="gaussian", help="Noise type: gaussian, normal (if possible)")
# For the general model we have to set up the potential
parser.add_argument('--optim', type=str, default='adam', help='Which optimizer to use?')
parser.add_argument('--sig2', type=float, default='0.0', help='Noise magnitude')
parser.add_argument('--mc_samples', type=int, default='1', help='amount of MC samples for noise')

parser.add_argument('--intrinsic_noise', type=float, default='0.0', help='amount of MC samples for noise')
parser.add_argument('--sig2_min', type=float, default='1e-09', help='amount of MC samples for noise')
parser.add_argument('--sig2_max', type=float, default='2.0', help='amount of MC samples for noise')
parser.add_argument('--sampling', action='store_false', help='Store false if fixed amoung of data to use.')
parser.add_argument('--N_samples', type=int, default=10000, help='How many samples to use (only if --sampling is False')
parser.add_argument('--N_epochs', type=int, default=500, help='How many times to run through samples (only if --sampling is False')


    

def compute_kl_pq_loss(model, batch):
    """ Compute BNAF eq 2 & 16:
    KL(p||q_fwd) where q_fwd is the forward flow transform (log_q_fwd = log_q_base + logdet), p is the target distribution.
    Returns the minimization objective for density estimation (NLL under the flow since the entropy of the target dist is fixed wrt the optimization) """
    z_, logdet_ = model(batch)
    # print('z_z_',z_)
    # print('logdet',logdet_)
    log_probs = torch.sum(model.base_dist.log_prob(z_)+ logdet_, dim=1) 
    return -log_probs.mean(0)  
    # return -log_probs/mc_samples   


# --------------------
# Validating
# --------------------
from torch.utils.data import Dataset

class NumpyValidationSet(Dataset):
    def __init__(self, x, device='cpu', dtype=torch.float):
        self.device = device
        self.dtype = dtype
        self.x = torch.from_numpy(x)

    def __getitem__(self, index):
        x = self.x[index, ...]
        return x.to(self.device,self.dtype)

    def __len__(self):
        return self.x.shape[0]
    
with torch.no_grad():   
    def validate_flow(model, val_loader, loss_fn):
        losses_val = 0
        
        for batch_data in val_loader:
            args.step += 1
            model.eval()
            
            batch_loss = loss_fn(model, batch_data)  
            
            losses_val += batch_loss.item()
        return losses_val/len(val_loader)

def evaluate_model(model, step ,args):
    first_batch = np.load(os.path.join(args.output_dir, 'first_batch.npy'))
    x = torch.from_numpy(first_batch[:,:,0]).to(args.device, torch.float)
    noise = torch.from_numpy(first_batch[:,:,1]).to(args.device, torch.float)    
    
    model.eval()
    z_, logdet_ = model(x)
    log_probs = torch.sum(model.base_dist.log_prob(z_)+ logdet_, dim=1) 
    np.save(os.path.join(args.output_dir, 'log_probs_' + str(step) +'.npy'),log_probs.detach().cpu().numpy()) 
     
    
    train_loader = torch.utils.data.DataLoader(x,batch_size=1, shuffle=False)
    
    # sing_values = np.zeros([args.batch_size,args.data_dim])
    
    eig_values = np.zeros([args.batch_size,args.data_dim,2])   
    
    for i, batch_data in enumerate(train_loader, 0):
        x = batch_data[0:1,:]
        
        x_ = torch.autograd.Variable(x)
        jac_ = torch.autograd.functional.jacobian(model.encode,x_)
        #print('jac_ should be 1x2x2 but is ', jac_.shape)
        jac_mat = jac_.reshape([args.data_dim,args.data_dim]) 
        # U,S_x,V = torch.svd(jac_mat)
        # sing_values[i,:] = S_x.detach().cpu().numpy()
        
        L,_ = torch.eig(jac_mat)
        eig_values[i,:,:] = L.detach().cpu().numpy()
        
    # np.save(os.path.join(args.output_dir, 'sing_values_' + str(np.int(os.getenv('SLURM_ARRAY_TASK_ID'))-1) +'.npy'),sing_values) 
    np.save(os.path.join(args.output_dir, 'eig_values_' + str(step) +'.npy'),eig_values) 



# --------------------
# Training
# --------------------

def train_flow(model, simulator, loss_fn, optimizer, scheduler, args, double_precision=False):
    losses = []
    best_loss = np.inf
    dtype = torch.double if double_precision else torch.float
    
    validation_set = simulator.sample(1000)
    validation_set = NumpyValidationSet(validation_set,device=args.device,dtype=dtype)
    
    val_loader = DataLoader(
    validation_set,
    shuffle=True,
    batch_size=args.batch_size,
    # pin_memory=self.run_on_gpu,
    #num_workers=n_workers,
            )
    
    if args.sampling == True:
    
        # with tqdm(total=args.n_gradient_steps, desc='Start step {}; Training for {} steps'.format(args.step,args.n_gradient_steps)) as pbar:
        for step in range(args.step+1,args.n_gradient_steps):
            args.step += 1
            
            model.train()
            batch = simulator.sample_and_noise(args.batch_size,sig2=args.sig2) 
            
            ## save data of first batch for estimate d
            if step == 1:
                np.save(os.path.join(args.output_dir, 'first_batch.npy'), batch)
                
                    
            # pdb.set_trace()
            x = torch.from_numpy(batch[:,:,0]).to(args.device, dtype)
            noise = torch.from_numpy(batch[:,:,1]).to(args.device, dtype)
            
            
            if args.sig2 >0:
                x_tilde = x + noise
            else:
                x_tilde = x
            
            loss = loss_fn(model, x_tilde)  
                   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if args.optim == 'adam':
                scheduler.step(loss) #x
            
            # pbar.set_postfix(loss = '{:.3f}'.format(loss.item()))
            # pbar.update()
            
            if step %1000 == 0: #00
                # save model
                checkpoint = {'step': args.step,
                              'state_dict': model.state_dict(),
                              'optimizer' : optimizer.state_dict(),
                              'scheduler' : scheduler.state_dict()}
                torch.save(checkpoint , os.path.join(args.output_dir, 'checkpoint.pt'))
                
                # evaluate_model(model,step, args)
                
                # calculate_KS_stats(args,model,simulator)
            if step %10 == 0: 
                evaluate_model(model,step, args)
         
            if step %100 == 0 and step > args.n_gradient_steps - 5000:
                val_loss =  validate_flow(model, val_loader, loss_fn)
                    
                if val_loss < best_loss:
                    best_loss = val_loss
                    # save model
                    checkpoint = {'step': args.step,
                                  'state_dict': model.state_dict()}
                    torch.save(checkpoint , os.path.join(args.output_dir, 'checkpoint_best.pt'))

    elif args.sampling == False:
        training_set = simulator.sample_and_noise(args.N_samples,sig2=args.sig2)
        training_set = NumpyValidationSet(training_set,device=args.device,dtype=dtype)
        
        train_loader = DataLoader(
        training_set,
        shuffle=True,
        batch_size=args.batch_size,
        # pin_memory=self.run_on_gpu,
        #num_workers=n_workers,
                )
        
        for epoch in range(args.N_epochs):
            print('starting epoch ',epoch)
            for step, batch in enumerate(train_loader):
                model.train()
                
                if step+1 == 1:
                    np.save(os.path.join(args.output_dir, 'first_batch.npy'), batch.detach().cpu().numpy())
                  
                x = batch[:,:,0]
                noise = batch[:,:,1]   
                
                if args.sig2 >0:
                    x_tilde = x + noise
                else:
                    x_tilde = x
                    
                loss = loss_fn(model, x_tilde)  
                       
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()        
            
                if args.optim == 'adam':
                    scheduler.step(loss)
                
            # if step+1 %1000 == 0: #00
                # save model
            checkpoint = {'step': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer' : optimizer.state_dict(),
                          'scheduler' : scheduler.state_dict()}
            torch.save(checkpoint , os.path.join(args.output_dir, 'checkpoint.pt'))
                # calculate_KS_stats(args,model,simulator)
            
            if epoch+1 > args.N_epochs - 10:
                val_loss =  validate_flow(model, val_loader, loss_fn)
                if val_loss < best_loss:
                    best_loss = val_loss
                    # save model
                    checkpoint = {'step': epoch,
                                  'state_dict': model.state_dict()}
                    torch.save(checkpoint , os.path.join(args.output_dir, 'checkpoint_best.pt'))    
                    
            
                

if __name__ == '__main__':
    warnings.simplefilter("once")
    args = parser.parse_args()
    
    sig2_0 = args.sig2_min #1e-08
    sig2_1 = args.sig2_max # 2.0
    n_sigmas = 20
    delta = np.log( (sig2_1 / sig2_0)**(1/(n_sigmas-1)) )
    
    sigmas = np.zeros(n_sigmas) + sig2_0 
    for k in range(n_sigmas-1): 
        sigmas[k+1] = sigmas[k] * np.exp(delta)
        
    
    #sigmas = [0,1e-09, 5e-09, 1e-08, 5e-08, 1e-07, 5e-07,1e-06,5e-06,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.25,0.5,1.0,2.0, 3.0, 4.0,  6.0 , 8.0, 10.0 ]
    args.sig2 = sigmas[np.int(os.getenv('SLURM_ARRAY_TASK_ID'))-1]
    
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    logger.info("Hi!")
    
    param_string = 'sig2_'+str(np.int(os.getenv('SLURM_ARRAY_TASK_ID'))-1)+'_seed_'+str(args.seed)
    
    original_output_dir = os.path.join(args.output_dir, args.dataset)
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.latent_distribution,args.noise_type, str(args.data_dim), param_string) 
    if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir)

    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    model = BNAF(args.data_dim, args.n_hidden, args.hidden_dim).to(args.device)
    
    # save settings
    config = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__)) + \
             'Num trainable params: {:,.0f}\n\n'.format(sum(p.numel() for p in model.parameters())) + \
             'Model:\n{}'.format(model)
    config_path = os.path.join(args.output_dir, 'config.txt')
    if not os.path.exists(config_path):
        with open(config_path, 'a') as f:
            print(config, file=f)
    
    simulator = load_simulator(args)
    loss_fn = compute_kl_pq_loss

    if args.train:
        if args.optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_gradient_steps)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=args.lr_patience, verbose=True)
        elif args.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay, patience=args.lr_patience, verbose=True)
        else:
            raise RuntimeError('Invalid `optimizer`.')
        if args.restore_file:
            model, optimizer, scheduler, args.step = load_checkpoint(args.output_dir,model,optimizer,scheduler)
            # optim_checkpoint = torch.load(os.path.dirname(args.restore_file) + '/optim_checkpoint.pt', map_location=args.device)
            # optimizer.load_state_dict(optim_checkpoint['optimizer'])
            # scheduler.load_state_dict(optim_checkpoint['scheduler'])
        train_flow(model, simulator, loss_fn, optimizer, scheduler, args)
        
    logger.info("Final evaluation: ")
    model, optimizer, scheduler, args.step = load_checkpoint(args.output_dir,model,best=True)
    
    first_batch = np.load(os.path.join(args.output_dir, 'first_batch.npy'))
    x = torch.from_numpy(first_batch[:,:,0]).to(args.device, torch.float)
    noise = torch.from_numpy(first_batch[:,:,1]).to(args.device, torch.float)    
    
    model.eval()
    z_, logdet_ = model(x)
    log_probs = torch.sum(model.base_dist.log_prob(z_)+ logdet_, dim=1) 
    np.save(os.path.join(args.output_dir, 'log_probs_' + str(np.int(os.getenv('SLURM_ARRAY_TASK_ID'))-1) +'.npy'),log_probs.detach().cpu().numpy()) 
     
    
    train_loader = torch.utils.data.DataLoader(x,batch_size=1, shuffle=False)
    
    sing_values = np.zeros([args.batch_size,args.data_dim])
    
    eig_values = np.zeros([args.batch_size,args.data_dim,2])   
    
    for i, batch_data in enumerate(train_loader, 0):
        x = batch_data[0:1,:]
        

        
        x_ = torch.autograd.Variable(x)
        jac_ = torch.autograd.functional.jacobian(model.encode,x_)
        #print('jac_ should be 1x2x2 but is ', jac_.shape)
        jac_mat = jac_.reshape([args.data_dim,args.data_dim]) 
        U,S_x,V = torch.svd(jac_mat)
        sing_values[i,:] = S_x.detach().cpu().numpy()
        
        L,_ = torch.eig(jac_mat)
        eig_values[i,:,:] = L.detach().cpu().numpy()
        
    np.save(os.path.join(args.output_dir, 'sing_values_' + str(np.int(os.getenv('SLURM_ARRAY_TASK_ID'))-1) +'.npy'),sing_values) 
    np.save(os.path.join(args.output_dir, 'eig_values_' + str(np.int(os.getenv('SLURM_ARRAY_TASK_ID'))-1) +'.npy'),eig_values) 
        # np.save(os.path.join(args.output_dir, 'sing_values.npy'),sing_values)    


    # S = np.load(os.path.join(args.output_dir, 'sing_values.npy'))
    # np.save(os.path.join(args.output_dir, 'sing_values_' + str(np.int(os.getenv('SLURM_ARRAY_TASK_ID'))-1) +'.npy'),S) 

    logger.info("All done...have an amazing day!")
    # if args.plot:
    #     plot(model, potential_or_sampling_fn, args)
        
    # if args.calculate_KS:
    #     calculate_KS_stats(args,model,simulator)
    
