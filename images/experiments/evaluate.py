
""" Top-level script for evaluating models """

import numpy as np
import logging
import sys
import torch
import torch.distributions as D
import configargparse
import copy
import tempfile
import os
import time
import torch.nn as nn
from matplotlib import pyplot as plt

# append pytorch_fid to system variables
sys.path.append("/storage/homefs/ch19g182/anaconda3/lib/python3.8/site-packages/pytorch-fid-master/src/pytorch_fid")

sys.path.append("../")

from evaluation import mcmc, sq_maximum_mean_discrepancy
from datasets import load_simulator, SIMULATORS, IntractableLikelihoodError, DatasetNotAvailableError
from utils import create_filename, create_modelname, sum_except_batch, array_to_image_folder
from architectures import create_model
from architectures.create_model import ALGORITHMS
from torch.utils.data import DataLoader
import torch.distributions as D

logger = logging.getLogger(__name__)

try:
    from fid_score import calculate_fid_given_paths
except:
    logger.warning("Could not import fid_score, make sure that pytorch-fid is in the Python path")
    calculate_fid_given_paths = None


def parse_args():
    """ Parses command line arguments for the evaluation """

    parser = configargparse.ArgumentParser()

    # What what what
    parser.add_argument("--truth", action="store_true", help="Evaluate ground truth rather than learned model")
    parser.add_argument("--modelname", type=str, default=None, help="Model name. Algorithm, latent dimension, dataset, and run are prefixed automatically.")
    parser.add_argument("--algorithm", type=str, default="flow", choices=ALGORITHMS, help="Model: flow (AF), mf (FOM, M-flow), emf (Me-flow), pie (PIE), gamf (M-flow-OT)...")
    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=SIMULATORS, help="Dataset: spherical_gaussian, power, lhc, lhc40d, lhc2d, and some others")
    parser.add_argument("--OOD_dataset", type=str, default="gan2d", choices=SIMULATORS, help="Dataset: spherical_gaussian, power, lhc, lhc40d, lhc2d, and some others")
    parser.add_argument("-i", type=int, default=0, help="Run number")

    # Dataset details
    parser.add_argument("--truelatentdim", type=int, default=2, help="True manifold dimensionality (for datasets where that is variable)")
    parser.add_argument("--datadim", type=int, default=2, help="True data dimensionality (for datasets where that is variable)")
    parser.add_argument("--latent_dim", type=int, default=3, help="True latent dimensionality (for datasets where that is variable)") 
    parser.add_argument("--epsilon", type=float, default=0.3, help="Noise term (for datasets where that is variable)")
    #Inflation/Deflation addition
    parser.add_argument("--latent_distribution", type=str, default=None, help="Latent distribution (for datasets where that is variable)")

    # Model details
    parser.add_argument("--modellatentdim", type=int, default=2, help="Model manifold dimensionality")
    parser.add_argument("--specified", action="store_true", help="Prescribe manifold chart: FOM instead of M-flow")
    parser.add_argument("--outertransform", type=str, default="rq-coupling", help="Scalar base trf. for f: {affine | quadratic | rq}-{coupling | autoregressive}")
    parser.add_argument("--innertransform", type=str, default="affine-autoregressive", help="Scalar base trf. for h: {affine | quadratic | rq}-{coupling | autoregressive}")
    parser.add_argument("--lineartransform", type=str, default="permutation", help="Scalar linear trf: linear | permutation")
    parser.add_argument("--outerlayers", type=int, default=5, help="Number of transformations in f (not counting linear transformations)")
    parser.add_argument("--innerlayers", type=int, default=5, help="Number of transformations in h (not counting linear transformations)")
    parser.add_argument("--conditionalouter", action="store_true", help="If dataset is conditional, use this to make f conditional (otherwise only h is conditional)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Use dropout")
    parser.add_argument("--pieepsilon", type=float, default=0.01, help="PIE epsilon term")
    parser.add_argument("--pieclip", type=float, default=None, help="Clip v in p(v), in multiples of epsilon")
    parser.add_argument("--encoderblocks", type=int, default=3, help="Number of blocks in Me-flow / PAE encoder")
    parser.add_argument("--encoderhidden", type=int, default=102, help="Number of hidden units in Me-flow / PAE encoder")
    parser.add_argument("--splinerange", default=3.0, type=float, help="Spline boundaries")
    parser.add_argument("--splinebins", default=8, type=int, help="Number of spline bins")
    parser.add_argument("--levels", type=int, default=3, help="Number of levels in multi-scale architectures for image data (for outer transformation f)")
    parser.add_argument("--actnorm", action="store_true", help="Use actnorm in convolutional architecture")
    parser.add_argument("--batchnorm", action="store_true", help="Use batchnorm in ResNets")
    parser.add_argument("--linlayers", type=int, default=2, help="Number of linear layers before the projection for M-flow and PIE on image data")
    parser.add_argument("--linchannelfactor", type=int, default=2, help="Determines number of channels in linear trfs before the projection for M-flow and PIE on image data")
    parser.add_argument("--intermediatensf", action="store_true", help="Use NSF rather than linear layers before projecting (for M-flows and PIE on image data)")
    parser.add_argument("--decoderblocks", type=int, default=3, help="Number of blocks in PAE encoder")
    parser.add_argument("--decoderhidden", type=int, default=102, help="Number of hidden units in PAE encoder")
    # DNF additions
    parser.add_argument("--v_threshold", type=float, default=3., help="threshold for v component for setting p(x)=0")

    # Evaluation settings
    parser.add_argument("--samplesize", type=int, default=None, help="If not None, number of samples used for training")
    parser.add_argument("--evaluate", type=int, default=1000, help="Number of test samples to be evaluated")
    parser.add_argument("--generate", type=int, default=10000, help="Number of samples to be generated from model")
    parser.add_argument("--gridresolution", type=int, default=11, help="Grid ressolution (per axis) for likelihood eval")
    parser.add_argument("--observedsamples", type=int, default=20, help="Number of iid samples in synthetic 'observed' set for inference tasks")
    parser.add_argument("--slicesampler", action="store_true", help="Use slice sampler for MCMC")
    parser.add_argument("--mcmcstep", type=float, default=0.15, help="MCMC step size")
    parser.add_argument("--thin", type=int, default=1, help="MCMC thinning")
    parser.add_argument("--mcmcsamples", type=int, default=5000, help="Length of MCMC chain")
    parser.add_argument("--burnin", type=int, default=100, help="MCMC burn in")
    parser.add_argument("--evalbatchsize", type=int, default=100, help="Likelihood eval batch size")
    parser.add_argument("--evallabel", type=int, default=1, help="Which label to evaluate")
    
    parser.add_argument("--chain", type=int, default=0, help="MCMC chain")
    parser.add_argument("--trueparam", type=int, default=None, help="Index of true parameter point for inference tasks")
    # DNF additions
    parser.add_argument("--MAP_steps", type=int, default=0, help="Number of MAP steps to infere z (relevent for PAE on vector data only)")
    parser.add_argument("--only_fid", action="store_true", help="Only evaluating fid")
    parser.add_argument("--only_KS", action="store_true", help="Only calculating KS-stats (latent probs for thin_spiral)")
    parser.add_argument("--only_ood", action="store_true", help="Only OOD")
    parser.add_argument("--only_adversarial", action="store_true", help="Only adversarial")
    parser.add_argument("--estimate_d", action="store_true", help="Only adversarial")
    #inflation/deflation
    parser.add_argument("--plot_title", type=str, default="Histogram", help="Title of hisogram plot.")
    parser.add_argument("--label", default=1, type=int, help="Which mnist label to train")

   
    #for estimate d 
    parser.add_argument("--noise_type", type=str, default='uniform', help="Noise type for dequantization.")
    parser.add_argument("--noise_type_preprocess", type=str, default='uniform', help="Noise type for dequantization.")
    
    parser.add_argument("--sig2", type=float, default=0.1, help="Noise magnitude.")
    parser.add_argument("--sig2_preprocess", type=float, default=1.0, help="Noise magnitude for preprocessing")   
    parser.add_argument("--scale_factor", type=float, default=1, help="Scale factor for StyleGan images")    
    
    # Other settings
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--dir", type=str, default=r"D:\PROJECTS\DNF\github\DNF_playground", help="Base directory of repo")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")
    parser.add_argument("--skipgeneration", action="store_true", help="Skip generative mode eval")
    parser.add_argument("--skiplikelihood", action="store_true", help="Skip likelihood eval")
    parser.add_argument("--skipood", action="store_true", help="Skip OOD likelihood eval")
    parser.add_argument("--skipinference", action="store_true", help="Skip all inference tasks (likelihood eval and MCMC)")
    parser.add_argument("--skipmcmc", action="store_true", help="Skip MCMC")

    return parser.parse_args()

device = torch.device("cpu")
dtype = torch.float

#python evaluate_MNIST.py --modelname gaussian_deq_small --plot_title "Gaussian $\sigma^2=0.1$" --sig2 0 --estimate_d --dataset mnist --algorithm flow --modellatentdim 8 --outerlayers 5 --innerlayers 5 --lineartransform permutation --outertransform rq-coupling --innertransform rq-coupling --splinerange 10.0 --splinebins 11 --dir D:\PROJECTS\DNF\github\DNF_playground
def estimate_intrinsic_dim(args,model,simulator,filename):
    # test_data = simulator.load_dataset(train=True, dataset_dir=filename, limit_samplesize=None, joint_score=None, label=args.label)
    #test_data , _ = simulator.load_dataset(
    #    train=True, numpy=True, ood=False, dataset_dir=create_filename("dataset", None, args), true_param_id=None, joint_score=False, limit_samplesize=10,
    #)
    test_data = simulator.load_dataset(train=True, dataset_dir=create_filename("dataset", None, args), limit_samplesize=args.evaluate)
    estimate = []     
        
    train_loader = torch.utils.data.DataLoader(test_data,batch_size=1, shuffle=False)
    # next(iter(train_loader))
    
    resolution = torch.prod(torch.tensor(simulator.data_dim())) #simulator.data_dim() #torch.prod(torch.tensor(x_.shape))
    
    eig_values = np.zeros([args.evaluate,resolution,2])
    sing_values = np.zeros([args.evaluate,resolution])
    log_prob = []  
    
    import time
    for i, batch_data in enumerate(train_loader, 0):
        
        t = time.time() 
        # x = batch_data[0].to(torch.device("cuda")).float()
        x = batch_data[0][0:1,:].float().to(torch.device("cuda"))
        # print('x shapae',x.shape)
        print('x sum',torch.sum(x))
        #noise = np.sqrt(args.sig2) * torch.randn(x.shape,device=torch.device("cuda"),requires_grad = True)
        
        if args.algorithm == "flow":
            x_reco, log_prob_, _ = model(x, context=None)

        log_prob.append(log_prob_.detach().cpu().numpy() )
        
        x_ = torch.autograd.Variable(x)
        # noise_ = torch.autograd.Variable(noise)
        
        # x_tilde = x_ + noise_
        logger.info('Start calculating Jacobian')
        jac_ = torch.autograd.functional.jacobian(model.encode,x_)
        #np.save(create_filename("results", "jacobian", args), jac_.detach().cpu().numpy())

        # jac_mat = jac_[0].reshape([784,784])
        
        #import pdb
        #pdb.set_trace()
        
        jac_mat = jac_.reshape([resolution,resolution])        
        # e,_ = torch.eig(jac_mat)
        U,S,V = torch.svd(jac_mat)
        
        sing_values[i,:] = S.detach().cpu().numpy()
        # eig_values[i,:,:] = e.detach().cpu().numpy()
        
        #print('Estimated d ',np.sum(S_x * 255 <=1))
        # jac_ = torch.autograd.functional.jacobian(model.encode,x_tilde)
        # jac_mat_tilde = jac_[0].reshape([784,784])
        
        # U,S_tilde,V = torch.svd(jac_mat_tilde)
        # print('jac_x_tilde',S_tilde)

        #import pdb
        #pdb.set_trace() 
        
        elapsed = time.time() - t
        logger.info('Time needed to calculate singular valiues of 1 sample: %s sec',elapsed)
    
    log_prob = np.concatenate(log_prob, axis=0)  
    np.save(create_filename("results", "log_likelihoods_" + str(args.label), args), log_prob)
      
    np.save(create_filename("results", "singular_values", args), sing_values)
    # np.save(create_filename("results", "eigen_values", args), eig_values)
    
    # log_prob = np.concatenate(log_prob, axis=0) 
    # x_ = torch.autograd.Variable(x)
    # z_ = model(x_)

def sample_from_model(args, model, simulator, batchsize=200):
    """ Generate samples from model and store """

    logger.info("Sampling from model")

    x = model.sample(n=30, context=None).detach().cpu().numpy()
    # import pdb
    # pdb.set_trace() 
    x = np.clip(np.transpose(x, [0, 2, 3, 1]) / 256.0, 0.0, 1.0)

    fig = plt.figure(figsize=(6 * 3.0, 6 * 3.0))
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        plt.imshow(x[i])
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    # plt.savefig(filename.format(i_epoch + 1))
    # plt.close()
    
    fig.suptitle(args.plot_title,y=1.001,fontsize=35)
    # plt.show()   
    
    plt.savefig(create_filename("training_plot", "sample_grid", args))
    # x_gen_all = []
    # while len(x_gen_all) < args.generate:
    #     n = min(batchsize, args.generate - len(x_gen_all))

    #     if simulator.parameter_dim() is None:
    #         x_gen = model.sample(n=n).detach().cpu().numpy()

    #     elif args.trueparam is None:  # Sample from prior
    #         params = simulator.sample_from_prior(n)
    #         params = torch.tensor(params, dtype=torch.float)
    #         x_gen = model.sample(n=n, context=params).detach().cpu().numpy()

    #     else:
    #         params = simulator.default_parameters(true_param_id=args.trueparam)
    #         params = np.asarray([params for _ in range(n)])
    #         params = torch.tensor(params, dtype=torch.float)
    #         x_gen = model.sample(n=n, context=params).detach().cpu().numpy()

    #     x_gen_all += list(x_gen)

    # x_gen_all = np.array(x_gen_all)
    # np.save(create_filename("results", "samples", args), x_gen_all)
      
    # return x_gen_all

def test_ood(args, simulator, filename, model):
    #--------
    SMALL_SIZE = 10
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 30
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    #--------
    logger.info("OOD test")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    from scipy import stats


    for label in range(10):
        test_data = simulator.load_dataset(train=False, dataset_dir=filename, limit_samplesize=None, joint_score=None, label=label)
        #Fashions MNIST:
        # test_fashion =   torch.load(r'D:\PROJECTS\MNIST_data\FashionMNIST\processed\test.pt')
        # data, labels = data_[0], data_[1]
        # label_idx = ((labels == label).nonzero()).flatten() 
        # x = data[label_idx,:].flatten(start_dim=1,end_dim=-1)
        # label_idx = ((labels == label).nonzero()).flatten()  #get label
        # x_label = test_data[label_idx,:]
        log_prob = []     
        
        train_loader = torch.utils.data.DataLoader(test_data,batch_size=100, shuffle=True)
            
        import time
        for batch_data in train_loader:
            t = time.time() 
            x_ = batch_data[0].to(torch.device("cuda"))

            if args.algorithm == "flow":
                x_reco, log_prob_, _ = model(x_, context=None)
            elif args.algorithm == "dnf":
                x_reco, log_prob_, _ = model(x_, context=None, mode="dnf")
            else:
                x_reco, log_prob_, _ = model(x_, context=None, mode="mf") #"mf" if not args.skiplikelihood else "projection")

            log_prob.append(log_prob_.detach().cpu().numpy() )#- torch.linalg.norm(x_reco-x_).detach().cpu().numpy())
            elapsed = time.time() - t
            # logger.info('Time needed to evaluate 1 batch: %s sec',elapsed)
        log_prob = np.concatenate(log_prob, axis=0)        
        np.save(create_filename("results", "log_likelihoods_" + str(label), args), log_prob)
        
        gkde = stats.gaussian_kde(dataset=log_prob)
        x_min, x_max = log_prob.min(), log_prob.max()
        x = np.linspace(x_min-500,x_max+500, 100)
        y = gkde.evaluate(x)
        ax.plot(x,y,label=str(label))
        ax.fill_between(x,0,y,alpha=0.3)
        
        ax.legend(loc = 'upper left')
        
    plt.title(args.plot_title,fontsize=35)
    # plt.show()   
    plt.tight_layout()
    plt.savefig(create_filename("training_plot", "ood_histograms", args))

def semantic_latent_sampling(args, model, simulator, filename, label=1, batchsize=200):
    """ Generate samples from model and store """
    ##two possibilities: 
    ##    1. encode image and see impact when varying latent variables
    ##    2. sample random latent, vary latents, and decode to img-space 
    logger.info("Sampling from model")
    
    test_data = simulator.load_dataset(train=False, dataset_dir=filename, limit_samplesize=None, joint_score=None, label=label)

    x_test = test_data.__getitem__(1)[0].to('cuda')
    # import pdb
    # pdb.set_trace()     
    x_test = x_test.reshape([1,1,32,32])
    x_rec, x_grid = model.sample_latent_path(x=x_test,n=10)
    
    # import pdb
    # pdb.set_trace() 
    
    nrows = args.modellatentdim
    ncols = 10
    fig, ax = plt.subplots()
    fig = plt.figure(figsize=(ncols, nrows))
    
    
    for i in range(nrows):
        for j in range(ncols):
            x = x_grid[i,j,:].detach().cpu().numpy()
            x = np.clip(x / 256.0 , 0.0, 1.0)
            img = np.transpose(x, [1, 2, 0])
            ax = fig.add_subplot(nrows,ncols,ncols*i+j+1)  
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout() 
    fig.savefig(create_filename("training_plot", "semantic_latent_samples_"+str(label), args), bbox_inches = 'tight')
    
    fig, ax = plt.subplots()
    nrows = 1
    ncols = 2
    fig = plt.figure(figsize=(ncols, nrows))
    
    ax = fig.add_subplot(nrows,ncols,1)  
    x = x_test[0,:].detach().cpu().numpy()
    x = np.clip(x / 256.0 , 0.0, 1.0)
    img = np.transpose(x, [1, 2, 0])
    ax = fig.add_subplot(nrows,ncols,1)  
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])    
    
    ax = fig.add_subplot(nrows,ncols,2)  
    x = x_rec[0,:].detach().cpu().numpy()
    x = np.clip(x / 256.0 , 0.0, 1.0)
    img = np.transpose(x, [1, 2, 0])
    ax = fig.add_subplot(nrows,ncols,2)  
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])  

    plt.tight_layout() 
    fig.savefig(create_filename("training_plot", "semantic_latent_reconstruction_"+str(label), args), bbox_inches = 'tight')
    
def test_adversarial_robustness(args, simulator, filename, model, steps=10, label=1):
    def calculate_grad(x_,model):
        x_ = torch.autograd.Variable( x_)
        x_.requires_grad = True

        x_reco, log_prob_, _ = model(x_, context=None, mode="dnf")
        loss = log_prob_ + torch.linalg.norm( x_reco-x_)
        grad = torch.autograd.grad(loss, x_)
        return grad, loss
    
    alpha = 10 #stepsize
    fig = plt.figure(figsize=(10, 10))
    
    test_data = simulator.load_dataset(train=False, dataset_dir=filename, limit_samplesize=None, joint_score=None, label=label)
    
    x_test = test_data.__getitem__(1)[0].to('cuda')
    x_test = x_test.reshape([1,1,32,32]) 
#     x_test.requires_grad = True
# # output = model(input)
# # v = torch.autograd.Variable(torch.from_numpy(np.ones([1,10])))
#     # loss = log_prob_ - torch.linalg.norm(x_reco-x_test)
    
#     x_reco, log_prob_, _ = model(x_test, context=None, mode="dnf")
#     loss = log_prob_ - torch.linalg.norm( x_reco-x_test)

#     grad = torch.autograd.grad(loss, x_test)
    
    x_old = x_test
    for step in range(steps):
        grad, loss = calculate_grad(x_old,model)
        # import pdb
        # pdb.set_trace() 
        x_step = x_old + alpha * grad[0]
        # x_step = x_step
        x = np.clip(x_step[0,:].detach().cpu().numpy() / 256.0 , 0.0, 1.0)
        img = np.transpose(x, [1, 2, 0])
        ax = fig.add_subplot(1,steps,step+1)  
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        x_old = x_step
        value = np.round(loss.item(),3)
        plt.title(str(value))
        
    plt.tight_layout() 
    fig.savefig(create_filename("training_plot", "adversarial_gradient_attack", args), bbox_inches = 'tight')

# dim_labels = [11,7,13,13,12,12,11,10,13,11]
if __name__ == "__main__":
    # Parse args
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    # Silence PIL
    for key in logging.Logger.manager.loggerDict:
        if "PIL" in key:
            logging.getLogger(key).setLevel(logging.WARNING)

    logger.info("Hi!")
    logger.debug("Starting evaluate.py with arguments %s", args)

    args.i = np.int(os.getenv('SLURM_ARRAY_TASK_ID'))

    if args.sig2 >0:
        args.pieepsilon = np.sqrt(args.sig2) 
        
    # Model name
    if args.truth:
        create_modelname(args)
        logger.info("Evaluating simulator truth")
    else:
        create_modelname(args)
        logger.info("Evaluating model %s", args.modelname)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data set
    
    args.dataset = args.OOD_dataset
    simulator = load_simulator(args)
    
    # Load model
    if not args.truth:
        from collections import OrderedDict
        model = create_model(args, simulator=simulator)
        checkpoint = create_filename("model", None, args)  
        
        # import pdb
        # pdb.set_trace() 
            
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
        model.eval()
    else:
        model = None 
        
    if torch.cuda.is_available() and not args.truth: 
        device = torch.device("cuda")
        dtype = torch.float
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        model = model.to(device, dtype)    



    # sample_from_model(args, model, simulator)   
    filename = create_filename("dataset", None, args)
    
    if args.estimate_d:
        estimate_intrinsic_dim(args,model,simulator,filename)
        exit()
        
    if args.only_adversarial:
        test_adversarial_robustness(args,simulator, filename, model,label=1)
        exit()
    
    if args.only_ood:
        test_ood(args,simulator, filename, model)
    else:           
        semantic_latent_sampling(args, model, simulator, filename,label=1)
        semantic_latent_sampling(args, model, simulator, filename,label=0)
        test_ood(args,simulator, filename, model)
    
    # evaluate_test_samples(args, simulator, model=model, filename="model_{}_test")
    

    logger.info("All done! Have a nice day!")
