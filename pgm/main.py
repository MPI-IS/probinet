"""
Implementation of CRep, JointCRep, MTCOV, DynCRep and ACD algorithms.
"""

import argparse
import logging
from pathlib import Path
import time

import numpy as np
from sparse import COO

from .input.loader import import_data, import_data_mtcov
from .input.tools import log_and_raise_error
from .model.acd import AnomalyDetection
from .model.crep import CRep
from .model.dyncrep import DynCRep
from .model.jointcrep import JointCRep
from .model.mtcov import MTCOV


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to run the CRep, JointCRep, DynCRep, MTCOV and ACD algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Shared parser for common arguments
    shared_parser = argparse.ArgumentParser(add_help=False)

    # Define the subparsers
    subparsers = parser.add_subparsers(
        dest="algorithm", help="Choose the algorithm to run"
    )

    # Common arguments
    shared_parser.add_argument(
        "-f",
        "--files",
        type=str,
        default="data/input",
        help="Path to the input " "files",
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
        type=bool,
        default=False,
        help="Flag to force a dense transformation in input",
    )
    shared_parser.add_argument(
        "--assortative", type=bool, default=False, help="Flag for assortative mixing"
    )
    shared_parser.add_argument(
        "-u",
        "--undirected",
        type=bool,
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
        "--flag_conv",
        type=str,
        choices=["log", "deltas"],
        default="log",
        help="Flag for convergence",
    )
    shared_parser.add_argument(
        "--plot_loglikelihood",
        type=bool,
        default=False,
        help="Flag to plot the log-likelihood",
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
        default="output",
        help="Path of the output folder",
    )
    shared_parser.add_argument(
        "--end_file", type=str, default="", help="Suffix for the output file"
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
        "CRep", help="Run the CRep algorithm", parents=[shared_parser]
    )
    crep_parser.add_argument(
        "-K", "--K", type=int, default=3, help="Number of communities"
    )
    crep_parser.add_argument(
        "-nr", "--num_realizations", type=int, default=5, help="Number of realizations"
    )
    crep_parser.add_argument("--rseed", type=int, default=0, help="Random seed")
    crep_parser.add_argument(
        "--mask", type=str, default=None, help="Mask for the data"
    )  # TODO: Rethink this. Not sure if the mask can be passed as CLI arg.
    crep_parser.add_argument(
        "--constrained",
        type=bool,
        default=False,
        help="Flag for constrained optimization",
    )
    crep_parser.add_argument(
        "--fix_eta", type=bool, default=False, help="Flag to fix eta"
    )
    crep_parser.add_argument(
        "--eta0", type=float, default=None, help="Initial eta value"
    )
    crep_parser.add_argument(
        "-A", "--adj_name", type=str, default="syn111.dat", help="Name of the network"
    )

    # JointCRep parser
    jointcrep_parser = subparsers.add_parser(
        "JointCRep", help="Run the JointCRep algorithm", parents=[shared_parser]
    )
    jointcrep_parser.add_argument(
        "-K", "--K", type=int, default=2, help="Number of communities"
    )
    jointcrep_parser.add_argument(
        "-nr", "--num_realizations", type=int, default=3, help="Number of realizations"
    )
    jointcrep_parser.add_argument("--rseed", type=int, default=0, help="Random seed")
    jointcrep_parser.add_argument(
        "--use_approximation",
        type=bool,
        default=False,
        help="Flag to use approximation",
    )
    jointcrep_parser.add_argument(
        "--fix_eta", type=bool, default=False, help="Flag to fix eta"
    )
    jointcrep_parser.add_argument(
        "--fix_w", type=bool, default=False, help="Flag to fix w"
    )
    jointcrep_parser.add_argument(
        "--fix_communities", type=bool, default=False, help="Flag to fix communities"
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
        "MTCOV", help="Run the MTCOV algorithm", parents=[shared_parser]
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
    mtcov_parser.add_argument("--rseed", type=int, default=0, help="Random seed")
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
        "-A", "--adj_name", type=str, default="adj.csv", help="Name of the network"
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
        "DynCRep", help="Run the DynCRep algorithm", parents=[shared_parser]
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
        "--mask", type=str, default=None, help="Mask for the data"
    )  # TODO: Rethink this. Not sure if the mask can be passed as CLI arg.
    dyncrep_parser.add_argument(
        "--fix_eta", type=bool, default=False, help="Flag to fix eta"
    )
    dyncrep_parser.add_argument(
        "--fix_beta", type=bool, default=False, help="Flag to fix beta"
    )
    dyncrep_parser.add_argument(
        "--eta0", type=float, default=None, help="Initial eta value"
    )
    dyncrep_parser.add_argument(
        "--beta0", type=float, default=0.25, help="Initial beta value"
    )
    dyncrep_parser.add_argument("--rseed", type=int, default=0, help="Random seed")
    dyncrep_parser.add_argument("--ag", type=float, default=1.1, help="Parameter ag")
    dyncrep_parser.add_argument("--bg", type=float, default=0.5, help="Parameter bg")
    dyncrep_parser.add_argument(
        "--temporal",
        type=bool,
        default=True,
        help="Flag to use non-temporal version of DynCRep",
    )
    dyncrep_parser.add_argument(
        "--fix_communities", type=bool, default=False, help="Flag to fix communities"
    )
    dyncrep_parser.add_argument(
        "--fix_w", type=bool, default=False, help="Flag to fix w"
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
        "--constraintU", type=bool, default=False, help="Flag for " "constraint U"
    )
    dyncrep_parser.add_argument(
        "-A",
        "--adj_name",
        type=str,
        default="synthetic_data_for_DynCRep.dat",
        help="Name of the network",
    )

    # ACD parser
    acd_parser = subparsers.add_parser(
        "ACD", help="Run the ACD algorithm", parents=[shared_parser]
    )
    acd_parser.add_argument(
        "-K", "--K", type=int, default=3, help="Number of communities"
    )
    acd_parser.add_argument(
        "-nr", "--num_realizations", type=int, default=1, help="Number of realizations"
    )
    acd_parser.add_argument(
        "--mask", type=str, default=None, help="Mask for the data"
    )  # TODO: Rethink this. Not sure if the mask can be passed as CLI arg.
    acd_parser.add_argument("--rseed", type=int, default=0, help="Random seed")
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
        type=bool,
        default=False,
        help="Flag for constrained optimization",
    )
    acd_parser.add_argument(
        "--fix_communities", type=bool, default=False, help="Flag to fix communities"
    )
    acd_parser.add_argument(
        "--fix_pibr", type=bool, default=False, help="Flag to fix pibr"
    )
    acd_parser.add_argument(
        "--fix_mupr", type=bool, default=False, help="Flag to fix mupr"
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
    # Step 1: Parse the command-line arguments
    args = parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="*** [%(levelname)s][%(asctime)s][%(module)s] %(message)s",
    )

    # Print all the args used in alphabetical order
    logging.debug("Arguments used:")
    for arg in sorted(vars(args)):
        logging.debug("%s: %s", arg, getattr(args, arg))

    # Step 2: Import the data

    # Set default values
    binary = True
    noselfloop = True
    undirected = False

    # Change values if the algorithm is not 'CRep'
    if args.algorithm != "CRep":
        binary = args.binary
        noselfloop = args.noselfloop
        undirected = args.undirected

    if args.algorithm != "MTCOV":
        if args.algorithm == "DynCRep":
            binary = True  # exactly this in source
            args.force_dense = True  # exactly this in source

        network = args.files + "/" + args.adj_name
        if args.algorithm != "ACD":
            A, B, B_T, data_T_vals = import_data(
                network,
                ego=args.ego,
                alter=args.alter,
                undirected=undirected,
                force_dense=args.force_dense,
                noselfloop=noselfloop,
                binary=binary,
                header=0,
            )
        else:
            A, B, B_T, data_T_vals = import_data(
                network,
                header=0,
            )
        logging.debug("Data looks like this: %s", B)
        logging.debug("Data loaded successfully from %s", network)
        nodes = A[0].nodes()
        Xs = None

        if args.algorithm == "DynCRep":
            if args.T is None:
                args.T = B.shape[0] - 1
            logging.debug("T = %s", args.T)

    else:
        A, B, X, nodes = import_data_mtcov(
            args.files,
            adj_name=args.adj_name,
            cov_name=args.cov_name,
            ego=args.ego,
            alter=args.alter,
            egoX=args.egoX,
            attr_name=args.attr_name,
            undirected=undirected,
            force_dense=args.force_dense,
            noselfloop=True,
        )
        Xs = np.array(X)
        B_T = None
        data_T_vals = None

        valid_types = [np.ndarray, COO]
        assert any(isinstance(B, vt) for vt in valid_types)
        logging.debug("Data loaded successfully from %s", args.files)

    def fit_model(model, algorithm):
        """
        Fit the model to the data.
        """
        # Define main parser arguments
        main_args = ["algorithm", "debug"]
        # Define the args that set numerical parameters
        numerical_args = [
            "num_realizations",
            "convergence_tol",
            "plot_loglikelihood",
            "flag_conv",
        ]
        # Define the args that are related to data loading
        data_loading_args = [
            "ego",
            "egoX",
            "alter",
            "attr_name",
            "cov_name",
            "force_dense",
            "noselfloop",
            "binary",
            "adj_name",
        ]
        filtered_args = {
            k: v
            for k, v in vars(args).items()
            if k not in data_loading_args
            and k not in numerical_args
            and k not in main_args
        }

        if algorithm == "CRep":
            model.fit(
                data=B,
                data_T=B_T,
                data_T_vals=data_T_vals,
                nodes=nodes,
                **filtered_args,
            )
        elif algorithm == "JointCRep":
            model.fit(
                data=B,
                data_T=B_T,
                data_T_vals=data_T_vals,
                nodes=nodes,
                **filtered_args,
            )
        elif algorithm == "DynCRep":
            model.fit(data=B, nodes=nodes, **filtered_args)
        elif algorithm == "ACD":
            model.fit(data=B, nodes=nodes, **filtered_args)
        elif algorithm == "MTCOV":
            model.fit(data=B, data_X=Xs, nodes=nodes, **filtered_args)

    # Step 5: Run the algorithm

    logging.info("Setting: K = %s", args.K)
    if args.algorithm == "MTCOV":
        logging.info("gamma = %s", args.gamma)
    logging.info("### Running %s ###", args.algorithm)
    if "DynCRep" in args.algorithm:
        logging.info("### Version: %s ###", "w-DYN" if args.temporal else "w-STATIC")

    # Map algorithm names to their classes
    algorithm_classes = {
        "CRep": CRep,
        "JointCRep": JointCRep,
        "MTCOV": MTCOV,
        "DynCRep": DynCRep,
        "ACD": AnomalyDetection,
    }

    # Create the model
    if args.algorithm in algorithm_classes:
        model = algorithm_classes[args.algorithm](
            flag_conv=args.flag_conv, num_realizations=args.num_realizations
        )

    else:
        log_and_raise_error(ValueError, "Algorithm not implemented.")

    # Time the execution
    time_start = time.time()

    # Fit the model to the data
    fit_model(model, args.algorithm)

    # Print the time elapsed
    logging.info("Time elapsed: %.2f seconds.", time.time() - time_start)
