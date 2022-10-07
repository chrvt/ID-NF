import numpy as np
import os
import logging
import torch

from .base import BaseSimulator, IntractableLikelihoodError
from .utils import NumpyDataset

logger = logging.getLogger(__name__)


class MNISTSimulator(BaseSimulator):
    """ MNIST in vector format """

    def __init__(self, latent_dim=8, data_dim=784, noise_type = 'uniform', sig2 = 1.):
        super().__init__()

        self._latent_dim = latent_dim
        self._data_dim = data_dim
        self._noise_type = noise_type
        self._sig2 = sig2

    def latent_dist(self):
        return self._latent_distribution

    def is_image(self):
        return False

    def data_dim(self):
        return self._data_dim

    def latent_dim(self):
        return self._latent_dim
    
    def parameter_dim(self):
        return None
    
    def _preprocess(self, img, num_bits=8):
        # img = np.copy(x)
        
        if img.dtype == torch.uint8:
            img = img.float()  # Already in [0,255]
        else:
            img = img * 255.0  # [0,1] -> [0,255]

        if num_bits != 8:
            img = torch.floor(img / 2 ** (8 - num_bits))  # [0, 255] -> [0, num_bins - 1]

        # Uniform dequantization.
        if self._noise_type == 'uniform':
            img = img + torch.rand_like(img)
        elif self._noise_type == 'gaussian':
            img = img + np.sqrt(self._sig2) * torch.rand(img.shape)
        
        # #map to [0,1]
        img -= img.min(1, keepdim=True)[0]
        img /= img.max(1, keepdim=True)[0]
        return img
    
    def load_dataset(self, train, dataset_dir, label=1, numpy=True, limit_samplesize=None, true_param_id=0, joint_score=False, ood=False, paramscan=False, run=0):
        tag = "train" if train else "test_"+str(label)
                
        path_to_data = r"{}/x_{}.npy".format(dataset_dir, tag)
        
        
        if os.path.exists(path_to_data):
            x = np.load(os.path.normpath(path_to_data))

        else:
            if train:		
                data_ = torch.load(r'/storage/homefs/ch19g182/Python/DNF_playground/experiments/data/MNIST/training.pt')
                data, labels = data_[0], data_[1]  # torch.unsqueeze(data_[0], dim=1) for adding channel dim
                label_idx = ((labels == label).nonzero()).flatten()  #get label
                x = data[label_idx,:].flatten(start_dim=1,end_dim=-1).numpy()
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)
                x_save_path = r"{}/{}.npy".format(dataset_dir, 'x_train_'+str(label))
                np.save(os.path.normpath(x_save_path),x)
            else:        
                data_ = torch.load(r'/storage/homefs/ch19g182/Python/DNF_playground/experiments/data/MNIST/test.pt')
                data, labels = data_[0], data_[1]
                label_idx = ((labels == label).nonzero()).flatten()  #get label
                x = data[label_idx,:].flatten(start_dim=1,end_dim=-1).numpy()
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)
                x_save_path = r"{}/{}.npy".format(dataset_dir, 'x_test_'+str(label))
                np.save(os.path.normpath(x_save_path),x)
        
        if limit_samplesize:
            x = x[0:limit_samplesize,:]
        
        #preprocess
        x = self._preprocess(torch.from_numpy(x), num_bits=8)
        params = np.ones(x.shape[0])
        # import pdb
        # pdb.set_trace() 
        return NumpyDataset(x.numpy(), params)
