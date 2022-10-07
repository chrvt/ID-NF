import torch
import numpy as np
import logging
from matplotlib import pyplot as plt
import os

logger = logging.getLogger(__name__)


def plt_latent_distribution(output_dir,simulator,model,i_epoch=0,n_grid=100,dtype=torch.float,device=torch.device("cpu")):
    """ Plots latent density """
    if simulator.latent_dim() == 2:
        model.eval()
        data_grid, latent, true_probs, jacobians, multiplier = simulator.generate_grid(n_grid,mode='data_space')
        
        u_, v_ = latent[0], latent[1]
        
        data_grid = torch.from_numpy(data_grid).to(device, dtype)  
        
        logprobs = []
        for xx_k in data_grid.split(200, dim=0):
            with torch.no_grad():
                z_k, logdet_k = model(xx_k)	
            logprobs += [torch.sum(model.base_dist.log_prob(z_k)+ logdet_k, dim=1) ]
        log_prob = torch.cat(logprobs, 0)     
        
        # z_, logdet_ = model(data_grid)
        # log_prob = torch.sum(model.base_dist.log_prob(z_)+ logdet_, dim=1) 
        
        # import pdb
        # pdb.set_trace()
            
        probs = torch.exp(log_prob).view(latent[1].shape[0],latent[0].shape[0]).detach().cpu().numpy() * jacobians # * np.sqrt(2*np.pi*sig2)  <--constant irrelevant for plotting
        
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(121)                    #latent[1].shape[0]-1,latent[1].shape[0]-1
        
        # if simulator.manifold =='sphere':
        #     u,v = np.meshgrid(v_, u_)
        # else:   
            
        u,v = np.meshgrid(u_, v_)
        
        ax.pcolormesh(u,v, true_probs.reshape([len(latent[1]),len(latent[0])]), cmap = plt.cm.jet, shading='auto')
        
        ax = fig.add_subplot(122)
        ax.pcolormesh(u,v, probs, cmap = plt.cm.jet, shading='auto')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latent_density_step_{}.pdf'.format(i_epoch)))
        plt.close()
    elif simulator.latent_dim() == 1:
        data_, latent, true_probs, jacobians, multiplier = simulator.generate_grid(n_grid,mode='data_space')
        
        #model probs
        model.eval()
        # logprobs = []
        # xx = torch.tensor(data_).to(device, dtype)  
        # for xx_k in xx.split(200, dim=0):
        #     with torch.no_grad():
        #         z_k, logdet_k = model(xx_k)	
        #         logprobs += [torch.sum(model.base_dist.log_prob(z_k)+ logdet_k, dim=1) ]
        # logprobs = torch.cat(logprobs, 0) 
        
        data_ = torch.from_numpy(data_).to(device, dtype)  
        
        logprobs = []
        for xx_k in data_.split(200, dim=0):
            with torch.no_grad():
                z_k, logdet_k = model(xx_k)	
            logprobs += [torch.sum(model.base_dist.log_prob(z_k)+ logdet_k, dim=1) ]
        log_prob = torch.cat(logprobs, 0)     
                    
        model_probs = torch.exp(log_prob).detach().cpu().numpy() * np.sqrt(jacobians) * np.sqrt(2*np.pi*0.01)  
        
        # import pdb
        # pdb.set_trace()
        
        fig = plt.figure(figsize=(4., 6.))
        ax = fig.add_subplot(1,1,1) 
        ax.plot(latent,true_probs,label='original')
        # line ='--'
        ax.plot(latent,model_probs,label='model',linestyle = '--')
        
        # ax.pcolormesh(u,v, probs, cmap = plt.cm.jet, shading='auto')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latent_density_step_{}.pdf'.format(i_epoch)))
        plt.close()

def plt_latent_fom(output_dir,simulator,model,i_epoch=0,n_grid=100,dtype=torch.float,device=torch.device("cpu")):
    """ Plots latent density """
    if simulator.latent_dim() == 2:
        model.eval()
        data_grid, latent, true_probs, jacobians, multiplier = simulator.generate_grid(n_grid,mode='data_space')
        
        u_, v_ = latent[0], latent[1]
        u,v = np.meshgrid(u_, v_)
        z_grid = np.stack((u.flatten(), v.flatten()), axis=1)
        # import pdb
        # pdb.set_trace() 
        # z_ = np.concatenate([u.reshape([n_grid,1]),v_.reshape([n_grid,1])],axis=1)
        z_grid = torch.from_numpy(z_grid).to(device, dtype) #.reshape([len(latent),1]) 
        
        logprobs = []
        for xx_k in z_grid.split(200, dim=0):
            with torch.no_grad():
                z_k, logdet_k = model(xx_k)	
            logprobs += [torch.sum(model.base_dist.log_prob(z_k)+ logdet_k, dim=1) ]
        log_prob = torch.cat(logprobs, 0)     
        
        # z_, logdet_ = model(data_grid)
        # log_prob = torch.sum(model.base_dist.log_prob(z_)+ logdet_, dim=1) 
        
        # import pdb
        # pdb.set_trace()
            
        probs = torch.exp(log_prob).view(latent[1].shape[0],latent[0].shape[0]).detach().cpu().numpy() #  * jacobians # * np.sqrt(2*np.pi*sig2)  <--constant irrelevant for plotting
        
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(121)                    #latent[1].shape[0]-1,latent[1].shape[0]-1
        
        # if simulator.manifold =='sphere':
        #     u,v = np.meshgrid(v_, u_)
        # else:   
            
        u,v = np.meshgrid(u_, v_)
        
        ax.pcolormesh(u,v, true_probs.reshape([len(latent[1]),len(latent[0])]), cmap = plt.cm.jet, shading='auto')
        
        ax = fig.add_subplot(122)
        ax.pcolormesh(u,v, probs, cmap = plt.cm.jet, shading='auto')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latent_density_step_{}.pdf'.format(i_epoch)))
        plt.close()
    elif simulator.latent_dim() == 1:
        data_, latent, true_probs, jacobians, multiplier = simulator.generate_grid(n_grid,mode='data_space')
        
        #model probs
        model.eval()
        # logprobs = []
        # xx = torch.tensor(data_).to(device, dtype)  
        # for xx_k in xx.split(200, dim=0):
        #     with torch.no_grad():
        #         z_k, logdet_k = model(xx_k)	
        #         logprobs += [torch.sum(model.base_dist.log_prob(z_k)+ logdet_k, dim=1) ]
        # logprobs = torch.cat(logprobs, 0) 
        
        z_grid = torch.from_numpy(latent).to(device, dtype).reshape([len(latent),1]) 
        
        logprobs = []
        for xx_k in z_grid.split(200, dim=0):
            with torch.no_grad():
                z_k, logdet_k = model(xx_k)	
            logprobs += [torch.sum(model.base_dist.log_prob(z_k)+ logdet_k, dim=1) ]
        log_prob = torch.cat(logprobs, 0)     
        

        model_probs = torch.exp(log_prob).detach().cpu().numpy() #* np.sqrt(jacobians) #* np.sqrt(2*np.pi*0.01)  
        
        import pdb
        pdb.set_trace()
        
        fig = plt.figure(figsize=(4., 6.))
        ax = fig.add_subplot(1,1,1) 
        ax.plot(latent,true_probs,label='original')
        # line ='--'
        ax.plot(latent,model_probs,label='model',linestyle = '--')
        
        # ax.pcolormesh(u,v, probs, cmap = plt.cm.jet, shading='auto')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latent_density_step_{}.pdf'.format(i_epoch)))
        plt.close()    