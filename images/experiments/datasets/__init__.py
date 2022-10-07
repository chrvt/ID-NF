import logging

from .base import IntractableLikelihoodError, DatasetNotAvailableError
from .spherical_simulator import SphericalGaussianSimulator
from .conditional_spherical_simulator import ConditionalSphericalGaussianSimulator
from .images import ImageNetLoader, CelebALoader, FFHQStyleGAN2DLoader, IMDBLoader, FFHQStyleGAN64DLoader, MNISTLoader, IsomapLoader
from .collider import WBFLoader, WBF2DLoader, WBF40DLoader
from .polynomial_surface_simulator import PolynomialSurfaceSimulator
from .lorenz import LorenzSimulator
from .thin_spiral import ThinSpiralSimulator
from .thin_disk import ThinDiskSimulator
from .d_sphere_simulator import d_SphereSimulator
from .d_gaussian_simulator import d_GaussianSimulator
from .utils import NumpyDataset
# from .mnist_simulator import MNISTSimulator

logger = logging.getLogger(__name__)


SIMULATORS = ["isomap", "d_sphere","d_gaussian", "mnist","power", "spherical_gaussian", "thin_spiral", "conditional_spherical_gaussian", "lhc", "lhc40d", "lhc2d", "imagenet", "celeba", "gan2d", "gan64d", "lorenz", "imdb"]


def load_simulator(args):
    assert args.dataset in SIMULATORS
    if args.dataset == "power":
        simulator = PolynomialSurfaceSimulator(filename=args.dir + "/experiments/data/samples/power/manifold.npz")
    elif args.dataset == "spherical_gaussian":
        simulator = SphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "conditional_spherical_gaussian":
        simulator = ConditionalSphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "thin_spiral":    
        simulator = ThinSpiralSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "lhc":
        simulator = WBFLoader()
    elif args.dataset == "lhc2d":
        simulator = WBF2DLoader()
    elif args.dataset == "lhc40d":
        simulator = WBF40DLoader()
    elif args.dataset == "imagenet":
        simulator = ImageNetLoader()
    elif args.dataset == "celeba":
        simulator = CelebALoader(args.noise_type_preprocess, args.sig2, args.scale_factor)
    elif args.dataset == "gan2d":
        simulator = FFHQStyleGAN2DLoader(args.noise_type_preprocess, args.sig2, args.scale_factor )
    elif args.dataset == "gan64d":
        simulator = FFHQStyleGAN64DLoader(args.noise_type_preprocess, args.sig2, args.scale_factor ) #, sig2 = args.sig2 )
    elif args.dataset == "lorenz":
        simulator = LorenzSimulator()
    elif args.dataset == "imdb":
        simulator = IMDBLoader()
    elif args.dataset == "mnist":
        simulator = MNISTLoader(args.noise_type_preprocess, args.sig2, args.scale_factor, label=args.label )
    elif args.dataset == "isomap":
        simulator = IsomapLoader(args.noise_type, args.sig2, args.scale_factor )
    # elif args.dataset == "mnist":
    #     simulator = MNISTSimulator(latent_dim = args.truelatentdim, data_dim = 784, noise_type = args.noise_type, sig2 = args.sig2)

    elif args.dataset == "d_sphere":
        simulator = d_GaussianSimulator(data_dim = args.datadim, latent_dim=args.latent_dim, latent_distribution='uniform',noise_type=args.noise_type, epsilon=0.0)
    
    elif args.dataset == "d_sphere":
        simulator = d_GaussianSimulator(data_dim = args.datadim, latent_dim=args.latent_dim, latent_distribution='uniform',noise_type=args.noise_type, epsilon=0.0)

    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))

    args.datadim = simulator.data_dim()
    return simulator
