import logging

from .base import IntractableLikelihoodError, DatasetNotAvailableError
from .sphere_simulator import SphereSimulator
from .torus_simulator import TorusSimulator
from .hyperboloid_simulator import HyperboloidSimulator
from .polynomial_surface_simulator import PolynomialSurfaceSimulator
from .thin_spiral_simulator import ThinSpiralSimulator
from .two_thin_spirals_simulator import TwoThinSpiralsSimulator
from .swiss_roll_simulator import SwissRollSimulator
from .von_Mises_on_circle import VonMisesSimulator
from .spheroid_simulator import SpheRoidSimulator
from .stiefel_simulator import StiefelSimulator
from .circle_simulator import CircleSimulator
from .isomap_simulator import IsomapSimulator
from .d_sphere_simulator import d_SphereSimulator
from .d_gaussian_simulator import d_GaussianSimulator
from .lolipop_simulator import LolipopSimulator
from .utils import NumpyDataset

logger = logging.getLogger(__name__)


SIMULATORS = ["d_sphere","lolipop","d_gaussian","isomap","hyperboloid", "torus","sphere", "swiss_roll", "thin_spiral", "two_thin_spirals", "spheroid", "stiefel","circle"]


def load_simulator(args):
    assert args.dataset in SIMULATORS
    if args.dataset == "torus":
        simulator = TorusSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "hyperboloid":
        simulator = HyperboloidSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "thin_spiral":    
        simulator = ThinSpiralSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "two_thin_spirals":    
        simulator = TwoThinSpiralsSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "swiss_roll":    
        simulator = SwissRollSimulator(epsilon=args.sig2, latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "von_Mises_circle":
        simulator = VonMisesSimulator(args.truelatentdim, args.datadim, epsilon=args.sig2)
    elif args.dataset == "sphere":
        simulator = SphereSimulator(kappa=6.0,epsilon=args.intrinsic_noise,latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "spheroid":
        simulator = SpheRoidSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "stiefel":
        simulator = StiefelSimulator(latent_distribution=args.latent_distribution,noise_type=args.noise_type)
    elif args.dataset == "circle":
        simulator = CircleSimulator(data_dim = args.data_dim, latent_distribution=args.latent_distribution,noise_type=args.noise_type, epsilon=args.intrinsic_noise) 
    elif args.dataset == "isomap":
        simulator = IsomapSimulator(data_dim = args.data_dim, latent_distribution=args.latent_distribution,noise_type=args.noise_type, epsilon=args.intrinsic_noise)
    elif args.dataset == "d_sphere":
        simulator = d_SphereSimulator(data_dim = args.data_dim, latent_dim=args.latent_dim, latent_distribution=args.latent_distribution,noise_type=args.noise_type, epsilon=args.intrinsic_noise)
    elif args.dataset == "lolipop":
        simulator =  LolipopSimulator(data_dim = args.data_dim, latent_dim=args.latent_dim, latent_distribution=args.latent_distribution,noise_type=args.noise_type, epsilon=args.intrinsic_noise)
    elif args.dataset == "d_gaussian":
        simulator = d_GaussianSimulator(data_dim = args.data_dim, latent_dim=args.latent_dim, latent_distribution=args.latent_distribution,noise_type=args.noise_type, epsilon=args.intrinsic_noise)
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))

    # args.datadim = simulator.data_dim()
    return simulator
