"""
Main script to run the algorithms.
"""

import argparse
import dataclasses
import logging
import time
from pathlib import Path

import numpy as np

from .models.acd import AnomalyDetection
from .models.base import ModelBaseParameters
from .models.crep import CRep
from .models.dyncrep import DynCRep
from .models.jointcrep import JointCRep
from .models.mtcov import MTCOV
from .utils.tools import log_and_raise_error

PATH_TO_DATA = str(Path(__file__).parent / "data" / "input")

# Map algorithm names to their classes
ALGORITHM_CLASSES = {
    "CRep": CRep,
    "JointCRep": JointCRep,
    "MTCOV": MTCOV,
    "DynCRep": DynCRep,
    "ACD": AnomalyDetection,
}


def parse_args():
    """
    Parse the command-line arguments.

    Returns
    -------
    args : argparse.Namespace
           Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Script to run the CRep, JointCRep, DynCRep, MTCOV, and ACD algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Shared parser for common arguments
    shared_parser = argparse.ArgumentParser(
        add_help=False,
    )

    # Define the subparsers
    subparsers = parser.add_subparsers(
        dest="algorithm", help="Choose the algorithm to run"
    )

    # Common arguments
    shared_parser.add_argument(
        "-f",
        "--files",
        type=str,
        default=PATH_TO_DATA,
        help="Path to the input files",
    )
    shared_parser.add_argument(
        "-e", "--ego", type=str, default="source", help="Name of the source of the edge"
    )
    shared_parser.add_argument(
        "-t",
        "--alter",
        type=str,
        default="target",
        help="Name of the target of the edge",
    )
    shared_parser.add_argument(
        "--noselfloop", type=bool, default=True, help="Flag to remove self-loops"
    )
    shared_parser.add_argument(
        "--binary", type=bool, default=True, help="Flag to make the network binary"
    )
    shared_parser.add_argument(
        "--force_dense",
        action="store_true",
        default=False,
        help="Flag to force a dense transformation in input",
    )
    shared_parser.add_argument(
        "--assortative",
        action="store_true",
        default=False,
        help="Flag for assortative mixing",
    )
    shared_parser.add_argument(
        "-u",
        "--undirected",
        action="store_true",
        default=False,
        help="Flag to treat the network as undirected",
    )
    shared_parser.add_argument(
        "-tol",
        "--convergence_tol",
        type=float,
        default=None,
        help=("Tolerance used to determine convergence"),
    )
    shared_parser.add_argument(
        "-maxi",
        "--max_iter",
        type=int,
        default=None,
        help="Maximum number of iterations",
    )
    shared_parser.add_argument(
        "--flag_conv",
        type=str,
        choices=["log", "deltas"],
        default="log",
        help="Flag for convergence",
    )
    shared_parser.add_argument(
        "--plot_loglik",
        action="store_true",
        default=False,
        help="Flag to plot the log-likelihood",
    )
    shared_parser.add_argument(
        "--rseed", type=int, default=int(time.time() * 1000), help="Random seed"
    )
    shared_parser.add_argument(
        "--initialization", type=int, default=0, help="Initialization method"
    )
    shared_parser.add_argument(
        "--out_inference",
        action="store_true",
        help="Flag to save the inference results",
    )
    shared_parser.add_argument(
        "--out_folder",
        "-o",
        type=str,
        default="evaluation",
        help="Path of the evaluation folder",
    )
    shared_parser.add_argument(
        "--end_file", type=str, default="", help="Suffix for the evaluation file"
    )
    shared_parser.add_argument(
        "--debug",
        "-d",
        dest="debug",
        action="store_true",
        default=False,
        help="Enable debug mode",
    )

    # CRep parser
    crep_parser = subparsers.add_parser(
        "CRep",
        help="Run the CRep algorithm",
        parents=[shared_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    crep_parser.add_argument(
        "-K", "--K", type=int, default=3, help="Number of communities"
    )
    crep_parser.add_argument(
        "-nr", "--num_realizations", type=int, default=5, help="Number of realizations"
    )
    crep_parser.add_argument(
        "--constrained",
        action="store_true",
        default=False,
        help="Flag for constrained optimization",
    )
    crep_parser.add_argument(
        "--fix_eta", action="store_true", default=False, help="Flag to fix eta"
    )
    crep_parser.add_argument(
        "--eta0", type=float, default=None, help="Initial eta value"
    )
    crep_parser.add_argument(
        "-A", "--adj_name", type=str, default="syn111.dat", help="Name of the network"
    )

    # JointCRep parser
    jointcrep_parser = subparsers.add_parser(
        "JointCRep",
        help="Run the JointCRep algorithm",
        parents=[shared_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    jointcrep_parser.add_argument(
        "-K",
        "--K",
        type=int,
        default=2,
        help="Number of communities",
    )
    jointcrep_parser.add_argument(
        "-nr", "--num_realizations", type=int, default=3, help="Number of realizations"
    )
    jointcrep_parser.add_argument(
        "--use_approximation",
        type=bool,
        default=False,
        help="Flag to use approximation",
    )
    jointcrep_parser.add_argument(
        "--fix_eta", action="store_true", default=False, help="Flag to fix eta"
    )
    jointcrep_parser.add_argument(
        "--fix_w", action="store_true", default=False, help="Flag to fix w"
    )
    jointcrep_parser.add_argument(
        "--fix_communities",
        action="store_true",
        default=False,
        help="Flag to fix communities",
    )
    jointcrep_parser.add_argument(
        "--eta0", type=float, default=None, help="Initial eta value"
    )
    jointcrep_parser.add_argument(
        "-A",
        "--adj_name",
        type=str,
        default="synthetic_data.dat",
        help="Name of the network",
    )

    # MTCOV parser
    mtcov_parser = subparsers.add_parser(
        "MTCOV",
        help="Run the MTCOV algorithm",
        parents=[shared_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mtcov_parser.add_argument(
        "-K", "--K", type=int, default=2, help="Number of communities"
    )
    mtcov_parser.add_argument(
        "-nr", "--num_realizations", type=int, default=5, help="Number of realizations"
    )
    mtcov_parser.add_argument(
        "-x",
        "--egoX",
        type=str,
        default="Name",
        help="Name of the column with node labels",
    )
    mtcov_parser.add_argument(
        "-an",
        "--attr_name",
        type=str,
        default="Metadata",
        help="Name of the attribute to consider in MTCOV",
    )
    mtcov_parser.add_argument(
        "--gamma", type=float, default=0.5, help="Scaling hyper parameter in MTCOV"
    )
    mtcov_parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Size of the batch to use to compute the likelihood",
    )
    mtcov_parser.add_argument(
        "-A",
        "--adj_name",
        type=str,
        default="multilayer_network.csv",
        help="Name of the network",
    )
    mtcov_parser.add_argument(
        "-C",
        "--cov_name",
        type=str,
        default="X.csv",
        help="Name of the design matrix used in MTCOV",
    )

    # DynCRep parser
    dyncrep_parser = subparsers.add_parser(
        "DynCRep",
        help="Run the DynCRep algorithm",
        parents=[shared_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dyncrep_parser.add_argument(
        "-K", "--K", type=int, default=2, help="Number of communities"
    )
    dyncrep_parser.add_argument(
        "-nr", "--num_realizations", type=int, default=5, help="Number of realizations"
    )
    dyncrep_parser.add_argument(
        "-T", "--T", type=int, default=None, help="Number of time snapshots"
    )
    dyncrep_parser.add_argument(
        "--fix_eta", action="store_true", default=False, help="Flag to fix eta"
    )
    dyncrep_parser.add_argument(
        "--fix_beta", action="store_true", default=False, help="Flag to fix beta"
    )
    dyncrep_parser.add_argument(
        "--eta0", type=float, default=None, help="Initial eta value"
    )
    dyncrep_parser.add_argument(
        "--beta0", type=float, default=0.25, help="Initial beta value"
    )
    dyncrep_parser.add_argument("--ag", type=float, default=1.1, help="Parameter ag")
    dyncrep_parser.add_argument("--bg", type=float, default=0.5, help="Parameter bg")
    dyncrep_parser.add_argument(
        "--temporal",
        type=bool,
        default=True,
        help="Flag to use non-temporal version of DynCRep",
    )
    dyncrep_parser.add_argument(
        "--fix_communities",
        action="store_true",
        default=False,
        help="Flag to fix communities",
    )
    dyncrep_parser.add_argument(
        "--fix_w", action="store_true", default=False, help="Flag to fix w"
    )
    dyncrep_parser.add_argument(
        "-fdT",
        "--flag_data_T",
        type=str,
        default=0,
        help="Flag to use data_T. "
        "It is recommended to use 0, but in case it does not work, try 1.",
    )
    dyncrep_parser.add_argument(
        "--constrained",
        type=bool,
        default=False,
        help="Flag for constrained optimization",
    )
    dyncrep_parser.add_argument(
        "--constraintU",
        action="store_true",
        default=False,
        help="Flag for " "constraint U",
    )
    dyncrep_parser.add_argument(
        "-A",
        "--adj_name",
        type=str,
        default="dynamic_network.dat",
        help="Name of the network",
    )

    # ACD parser
    acd_parser = subparsers.add_parser(
        "ACD",
        help="Run the ACD algorithm",
        parents=[shared_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    acd_parser.add_argument(
        "-K", "--K", type=int, default=3, help="Number of communities"
    )
    acd_parser.add_argument(
        "-nr", "--num_realizations", type=int, default=1, help="Number of realizations"
    )
    acd_parser.add_argument("--ag", type=float, default=1.1, help="Parameter ag")
    acd_parser.add_argument("--bg", type=float, default=0.5, help="Parameter bg")
    acd_parser.add_argument("--pibr0", default=None, help="Anomaly parameter pi")
    acd_parser.add_argument("--mupr0", default=None, help="Prior mu")
    acd_parser.add_argument(
        "--flag_anomaly",
        action="store_false",
        default=True,
        help="Flag to detect anomalies",
    )
    acd_parser.add_argument(
        "--constrained",
        action="store_true",
        default=False,
        help="Flag for constrained optimization",
    )
    acd_parser.add_argument(
        "--fix_communities",
        action="store_true",
        default=False,
        help="Flag to fix communities",
    )
    acd_parser.add_argument(
        "--fix_pibr", action="store_true", default=False, help="Flag to fix pibr"
    )
    acd_parser.add_argument(
        "--fix_mupr", action="store_true", default=False, help="Flag to fix mupr"
    )
    acd_parser.add_argument(
        "-A",
        "--adj_name",
        type=str,
        default="synthetic_data_for_ACD.dat",
        help="Name of the network",
    )

    args = parser.parse_args()
    return args


def main():
    """
    Main function for CRep/JointCRep/MTCOV/DynCRep/ACD algorithms.
    """
    # Parse the command-line arguments
    args = parse_args()

    # Configure the logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="*** [%(levelname)s][%(asctime)s][%(module)s] %(message)s",
    )

    # Set the logging level for third-party packages to WARNING
    logging.getLogger("numba").setLevel(logging.WARNING)

    # Print all the args used in alphabetical order
    logging.debug("Arguments used:")
    for arg in sorted(vars(args)):
        logging.debug("%s: %s", arg, getattr(args, arg))

    # Define the args that set numerical parameters. These will be used to instantiate the models.
    numerical_args = [f.name for f in dataclasses.fields(ModelBaseParameters)]

    # Filter the numerical args
    numerical_args_dict = {
        k: v for k, v in vars(args).items() if k in numerical_args and v is not None
    }

    # Check if the algorithm is implemented
    if args.algorithm not in ALGORITHM_CLASSES:
        log_and_raise_error(ValueError, "Algorithm not implemented.")

    # Instantiate the models
    model = ALGORITHM_CLASSES[args.algorithm](**numerical_args_dict)

    # Get dictionary that contains the parameters needed to load the gdata
    data_kwargs = model.get_params_to_load_data(args)

    # Import graph data
    gdata = model.load_data(**data_kwargs)

    # Time the execution
    time_start = time.time()

    # Create the rng object and add it to the args
    args.rng = np.random.default_rng(seed=args.rseed)

    # Fit the models to the graph data using the fit args
    logging.info("### Running %s ###", model.__class__.__name__)
    logging.info("K = %s", args.K)
    model.fit(gdata, **vars(args))

    # Print the time elapsed
    logging.info("Time elapsed: %.2f seconds.", time.time() - time_start)
