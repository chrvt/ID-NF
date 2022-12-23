import logging
import numpy as np
import os
from torchvision import transforms as tvt
from scipy.stats import norm
import torch

from .utils import Preprocess, RandomHorizontalFlipTensor
from .base import BaseSimulator, DatasetNotAvailableError
from .utils import UnlabelledImageDataset, CSVLabelledImageDataset, LabelledImageDataset

logger = logging.getLogger(__name__)


class BaseImageLoader(BaseSimulator):
    def __init__(self, resolution, noise_type, sig2, scale_factor, n_bits=8, random_horizontal_flips=True, gdrive_file_ids=None, label=None):
        super().__init__()

        self.gdrive_file_ids = gdrive_file_ids
        self.resolution = resolution
        self.n_bits = n_bits
        self.random_horizontal_flips = random_horizontal_flips
        
        self.noise_type = noise_type
        self.sig2 = sig2 
        self.scale_factor = scale_factor
        self.label = label
        
    def is_image(self):
        return True

    def data_dim(self):
        return (3, self.resolution, self.resolution)

    def latent_dim(self):
        raise NotImplementedError

    def parameter_dim(self):
        return None

    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None, true_param_id=0, joint_score=False, ood=False, run=0):
        if ood:
            raise DatasetNotAvailableError()
        if joint_score:
            raise NotImplementedError("SCANDAL training not implemented for this dataset")

        # Download missing data
        self._download(dataset_dir)

        # Load data as numpy array
        x = np.load("{}/{}.npy".format(dataset_dir, "train" if train else "test"))
        # Optionally limit sample size
        if limit_samplesize is not None:
            logger.info("Only using %s of %s available samples", limit_samplesize, x.shape[0])
            x = x[75:limit_samplesize]

        if numpy:
            # 8-bit preprocessing, but without the dequantization
            assert self.n_bits == 8
            if np.max(x) <= 1.0:
                x = 0.5 + 255.0 * x
            assert 0.0 <= np.min(x)
            assert np.max(x) <= 256.0
            return x, None

        # Transforms
        if train and self.random_horizontal_flips:
            transform = tvt.Compose([RandomHorizontalFlipTensor(), Preprocess(self.n_bits,self.noise_type,self.sig2)])
        else:
            transform = Preprocess(self.n_bits,self.noise_type,self.sig2, self.scale_factor)

        # Dataset
        dataset = UnlabelledImageDataset(x, transform=transform)

        return dataset

    def sample(self, n, parameters=None):
        raise NotImplementedError

    def sample_ood(self, n, parameters=None):
        raise NotImplementedError

    def distance_from_manifold(self, x):
        raise NotImplementedError


class ImageNetLoader(BaseImageLoader):
    def __init__(self):
        super().__init__(
            resolution=64, n_bits=8, random_horizontal_flips=False, gdrive_file_ids={"train": "15AMmVSX-LDbP7LqC3R9Ns0RPbDI9301D", "test": "1Me8EhsSwWbQjQ91vRG1emkIOCgDKK4yC"},
        )


class CelebALoader(BaseImageLoader):
    def __init__(self, noise_type, sig2, scale_factor):
        super().__init__(
            resolution=int(64*scale_factor),
            noise_type=noise_type,
            sig2=sig2,
            scale_factor=scale_factor,
            n_bits=8,
            random_horizontal_flips=False,
            gdrive_file_ids={"train": "1bcaqMKWzJ-2ca7HCQrUPwN61lfk115TO", "test": "1WfE64z9FNgOnLliGshUDuCrGBfJSwf-t"},
        )


class FFHQStyleGAN2DLoader(BaseImageLoader):
    def __init__(self, noise_type, sig2, scale_factor):
        super().__init__(   
            resolution=int(64*scale_factor),
            noise_type=noise_type,
            sig2=sig2,
            scale_factor=scale_factor,
            n_bits=8,
            random_horizontal_flips=False,
            gdrive_file_ids={"grid": "1HZvfuOkaXqIMtPX_hpCp3zzHrL3JtEUb", "train": "1Plel_nOIYUu3E-KKDJ9-yVWPp5HcaGFo", "test": "17NOhkhctMkPWvLOzR5L0WOYxAFlUebjd"},
        )  # For the 2D demo we don't want random flips, as they would essentially create a second disjoint manifold

    def latent_dim(self):
        return 2


class FFHQStyleGAN64DLoader(BaseImageLoader):
    def __init__(self, noise_type, sig2, scale_factor):
        super().__init__(
            resolution=int(64*scale_factor),
            noise_type=noise_type,
            sig2=sig2,
            scale_factor=scale_factor,
            n_bits=8,
            random_horizontal_flips=False,
            gdrive_file_ids={
                "train": "1TsrUWSCcuRjOsUSpCn_HvlzC-Cz3HOKS",
                "test": "1gJuOOm9x5sNytuKmYqZ83-ieicfJsXj5",
                "params_train": "1MmIAfT2uvAC7fuC92KxNRQJUAUxsnXZr",
                "params_test": "1day5UUZBRxAfvQsmKxir8KL1RAYbsIY9",
            },
        )
        

    def latent_dim(self):
        return 64


    def sample_from_prior(self, n):
        return norm.rvs(size=(n, self.parameter_dim()))

    def evaluate_log_prior(self, parameters):
        parameters = parameters.reshape((-1, self.parameter_dim()))
        return np.sum(norm.logpdf(parameters), axis=1)
    

class IMDBLoader(BaseImageLoader):
    _AGES = np.linspace(18, 80, 63)
    _AGE_PROBS = np.array(
        [
            0.00835267561232834,
            0.011491477628632799,
            0.016188291676252388,
            0.018384086410445057,
            0.01868247760357705,
            0.022894121237936226,
            0.02284856533058783,
            0.0294200049655939,
            0.025796032536029027,
            0.03264080761512547,
            0.028750333127572487,
            0.03487076927982944,
            0.034478988476633235,
            0.03969741766339196,
            0.03612583452727774,
            0.033422091426150456,
            0.034677156673598754,
            0.03421476421401254,
            0.03554499670858569,
            0.030665959031572522,
            0.03445848831832646,
            0.034176041692766404,
            0.0275544905596771,
            0.026483926736989804,
            0.026841540609674707,
            0.024297243184266813,
            0.021582111106302433,
            0.023329180153113405,
            0.021431776612052728,
            0.0171335767537316,
            0.017021964780728028,
            0.015117727853565091,
            0.013846718038544854,
            0.013019878320171473,
            0.01211103796857098,
            0.011318365180708896,
            0.009049680994758794,
            0.009450572979424674,
            0.007758171021431777,
            0.00730944533405008,
            0.007550891642996577,
            0.007124943909289077,
            0.007402834944114291,
            0.006143214105931151,
            0.005416597383724241,
            0.005343707931966808,
            0.0042389771787682134,
            0.004261755132442412,
            0.003831251808000073,
            0.003243580603205769,
            0.0030841349274863847,
            0.0026262980586350083,
            0.003038579020137989,
            0.002225406073969127,
            0.0021935169388252497,
            0.0015602898266825504,
            0.0014851225795576978,
            0.0012710098150202382,
            0.001491955965659957,
            0.0012710098150202382,
            0.0010250079153389018,
            0.0009999521662972842,
            0.00073117231294175,
        ]
    )

    def __init__(self):
        super().__init__(resolution=64, n_bits=8, random_horizontal_flips=True)

    def parameter_dim(self):
        return 1

    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None, true_param_id=0, joint_score=False, ood=False, paramscan=False, run=0):
        if ood or paramscan:
            raise NotImplementedError()
        if joint_score:
            raise NotImplementedError("SCANDAL training not implemented for this dataset")
        if limit_samplesize is not None:
            raise NotImplementedError("IMDb dataset does not allow limiting the samplesize")
        if numpy:
            raise NotImplementedError("IMDb dataset cannot be loaded as numpy array for now")

        # Transforms
        if train and self.random_horizontal_flips:
            transform = tvt.Compose([RandomHorizontalFlipTensor(), Preprocess(self.n_bits)])
        else:
            transform = Preprocess(self.n_bits)

        # Dataset
        category = "train" if train else "test"
        return CSVLabelledImageDataset(
            f"{dataset_dir}/{category}.csv", label_key="age", filename_key="filename", root_dir=dataset_dir, image_transform=transform, label_transform=self.preprocess_params,
        )

    def sample_from_prior(self, n):
        parameters = np.random.choice(self._AGES, size=n, p=self._AGE_PROBS)
        parameters = self.preprocess_params(parameters)
        return parameters

    def evaluate_log_prior(self, parameters):
        parameters = self.preprocess_params(parameters, inverse=True)
        parameters = np.around(parameters, 0).astype(np.int)

        min_, max_ = np.min(self._AGES), np.max(self._AGES)
        idx = np.clip(parameters - min_, 0, max_ - min_).astype(np.int)
        probs = np.where(parameters < min_, 0, np.where(parameters > max_, 0, self._AGE_PROBS[idx]))
        return np.log(probs)

    @staticmethod
    def preprocess_params(x, inverse=False):
        _AGE_MEAN = 37.45325282219468
        _AGE_STD = 35.52156863651862
        x = np.copy(x).astype(np.float)
        if inverse:
            x *= _AGE_STD
            x += _AGE_MEAN
        else:
            x -= _AGE_MEAN
            x /= _AGE_STD
        return x

class MNISTLoader(BaseImageLoader):
    # def __init__(self, noise_type, sig2, scale_factor):
    #     super().__init__(
    #         resolution=int(64*scale_factor),
    #         noise_type=noise_type,
    #         sig2=sig2,
    #         scale_factor=scale_factor,
    #         n_bits=8,
    #         random_horizontal_flips=False,
    #         gdrive_file_ids={
    #             "x_train": "1DayM2MLczvmck9Nfdv_c5oYOD6GbRkGj",
    #             "x_test": "1gJuOOm9x5sNytuKmYqZ83-ieicfJsXj5",
    #             "params_train": "1MmIAfT2uvAC7fuC92KxNRQJUAUxsnXZr",
    #             "params_test": "1day5UUZBRxAfvQsmKxir8KL1RAYbsIY9",
    #         },
    #     )
    def __init__(self, noise_type, sig2, scale_factor, label):
        super().__init__(
            resolution=int(32*scale_factor),
            noise_type=noise_type,
            sig2=sig2,
            scale_factor=scale_factor,
            n_bits=8,
            random_horizontal_flips=False,
            label = label,
        )
        
    # def __init__(self, noise_type, sig2, scale_factor,):
    #     super().__init__( 
    #         resolution=int(64*scale_factor), noise_type=noise_type, sig2=sig2, scale_factor=scale_factor, n_bits=8, random_horizontal_flips=False, label = label,
    #     )
    
    def data_dim(self):
        return (1, self.resolution, self.resolution)    
    
    def load_dataset(self, train, dataset_dir, numpy=False, limit_samplesize=None, true_param_id=0, joint_score=False, ood=False, paramscan=False, run=0):
        # tag = "train" if train else "test_"+str(label)
        label = self.label
        path_to_data = r"{}/x_{}.npy".format(dataset_dir, label)
        
        
        if os.path.exists(path_to_data):
            if limit_samplesize:
                x = np.load(os.path.normpath(path_to_data))[0:limit_samplesize,:]
            else:
                x = np.load(os.path.normpath(path_to_data))
        else:
            if train:
                print('Loading and saving MNIST for label ',label)
                data_ = torch.load(r'/storage/homefs/ch19g182/Python/estimate_d/experiments/data/samples/MNIST/mnist_all.pt')
                data, labels = torch.unsqueeze(data_[0], dim=1), data_[1]  #.to(args.device,dtype).flatten(1,-1)
                if label >-1 :
                    label_idx = ((labels == label).nonzero()).flatten()  #get label
                    x = data[label_idx,:].numpy()
                    if not os.path.exists(dataset_dir):
                        os.makedirs(dataset_dir)
                    x_save_path = r"{}/{}{}.npy".format(dataset_dir,'x_',label)
                    np.save(os.path.normpath(x_save_path),x)
                    xb = torch.nn.functional.interpolate(torch.from_numpy(x), size=(32,32))
                else:
                    xb = torch.nn.functional.interpolate(data, size=(32,32))
            else:        
                data_ = torch.load(r'/storage/homefs/ch19g182/Python/estimate_d/experiments/data/samples/MNIST/mnist_all.pt')
                data, labels = torch.unsqueeze(data_[0], dim=1), data_[1]  #.to(args.device,dtype).flatten(1,-1)
                label_idx = ((labels == label).nonzero()).flatten()  #get label
                x = data[label_idx,:].numpy()
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)
                x_save_path = r"{}/{}.npy".format(dataset_dir, 'x_test_'+str(label))
                np.save(os.path.normpath(x_save_path),x)
        
        if numpy:
            # 8-bit preprocessing, but without the dequantization
            assert self.n_bits == 8
            if np.max(x) <= 1.0:
                x = 0.5 + 255.0 * x
            assert 0.0 <= np.min(x)
            assert np.max(x) <= 256.0
            return x, None
        
        # print('x stats -1 ' ,x[:,0,0,:])
        # Transforms  <<dequantization happens in Preprocess!!!>>
        if train and self.random_horizontal_flips:
            transform = tvt.Compose([RandomHorizontalFlipTensor(), Preprocess(self.n_bits)])
        else: 
            # print('self_noise_type :',self.noise_type)
            transform = Preprocess(self.n_bits,self.noise_type,self.sig2, self.scale_factor)
        # elif train:
        #     transform = Preprocess(self.n_bits,noise_type='gaussian',sigma=1)
        # else:
        #     transform = Preprocess(self.n_bits,self.noise_type,self.sig2, self.scale_factor)

        # Dataset

        print('xb preprocess mean ' ,torch.sum(xb[0,:]))
        dataset = UnlabelledImageDataset(xb.numpy(), transform=transform) #xb.numpy()

        return dataset
    
class IsomapLoader(BaseImageLoader):
    # def __init__(self, noise_type, sig2, scale_factor):
    #     super().__init__(
    #         resolution=int(64*scale_factor),
    #         noise_type=noise_type,
    #         sig2=sig2,
    #         scale_factor=scale_factor,
    #         n_bits=8,
    #         random_horizontal_flips=False,
    #         gdrive_file_ids={
    #             "x_train": "1DayM2MLczvmck9Nfdv_c5oYOD6GbRkGj",
    #             "x_test": "1gJuOOm9x5sNytuKmYqZ83-ieicfJsXj5",
    #             "params_train": "1MmIAfT2uvAC7fuC92KxNRQJUAUxsnXZr",
    #             "params_test": "1day5UUZBRxAfvQsmKxir8KL1RAYbsIY9",
    #         },
    #     )
        
    def __init__(self, noise_type, sig2, scale_factor):
        super().__init__(
            resolution=int(64*scale_factor),
            noise_type=noise_type,
            sig2=sig2,
            scale_factor=scale_factor,
            n_bits=8,
            random_horizontal_flips=False
        )
    
    def data_dim(self):
        return (1, self.resolution, self.resolution)    
    
    def load_dataset(self, train, dataset_dir, label=1, numpy=False, limit_samplesize=None, true_param_id=0, joint_score=False, ood=False, paramscan=False, run=0):
        tag = "train" if train else "test_"+str(label)
                
        path_to_data = r"{}/train.npy".format(dataset_dir)
        
        if os.path.exists(path_to_data):
            if limit_samplesize:
                x = np.load(os.path.normpath(path_to_data))[0:limit_samplesize,:]
            else:
                x = np.load(os.path.normpath(path_to_data))
        else:
            if train: 
                x = np.load(r'/storage/homefs/ch19g182/Python/estimate_d/experiments/data/samples/isomap/train.npy')
               # data, labels = torch.unsqueeze(data_[0], dim=1), data_[1]  #.to(args.device,dtype).flatten(1,-1)
               # label_idx = ((labels == label).nonzero()).flatten()  #get label
               # x = data[label_idx,:].numpy()
                # if not os.path.exists(dataset_dir):
                #     os.makedirs(dataset_dir)
                # x_save_path = r"{}/{}.npy".format(dataset_dir, 'x_train')
                # np.save(os.path.normpath(x_save_path),x.numpy())
            # else:        
            #     data_ = torch.load(r'/storage/homefs/ch19g182/Python/DNF_playground/experiments/data/MNIST/test.pt')
            #     data, labels = torch.unsqueeze(data_[0], dim=1), data_[1]  #.to(args.device,dtype).flatten(1,-1)
            #     label_idx = ((labels == label).nonzero()).flatten()  #get label
            #     x = data[label_idx,:].numpy()
            #     if not os.path.exists(dataset_dir):
            #         os.makedirs(dataset_dir)
            #     x_save_path = r"{}/{}.npy".format(dataset_dir, 'x_test_'+str(label))
            #     np.save(os.path.normpath(x_save_path),x)
        
        if numpy:
            # 8-bit preprocessing, but without the dequantization
            assert self.n_bits == 8
            if np.max(x) <= 1.0:
                x = 0.5 + 255.0 * x
            assert 0.0 <= np.min(x)
            assert np.max(x) <= 256.0
            return x, None
        
        # Transforms  <<dequantization happens in Preprocess!!!>>
        if train and self.random_horizontal_flips:
            transform = tvt.Compose([RandomHorizontalFlipTensor(), Preprocess(self.n_bits)])
        # elif train:
            # transform = Preprocess(self.n_bits,noise_type='gaussian',sigma=1)
        else:
            transform = Preprocess(self.n_bits,self.noise_type,self.sig2, self.scale_factor)

        # Dataset
        # xb = torch.nn.functional.interpolate(torch.from_numpy(x), size=(32,32))
        x = x.reshape([x.shape[0],1,64,64])
        dataset = UnlabelledImageDataset(x, transform=transform)  #data.numpy()

        return dataset
