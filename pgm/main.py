"""
Implementation of CRep, JointCRep, and MTCOV algorithm.
"""

import argparse
import logging
from pathlib import Path
import time

import numpy as np
import sktensor as skt

from .input.loader import import_data, import_data_mtcov
from .input.tools import log_and_raise_error
from .model.acd import AnomalyDetection
from .model.crep import CRep
from .model.dyncrep import DynCRep
from .model.jointcrep import JointCRep
from .model.mtcov import MTCOV


def parse_args():
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Script to run the CRep, JointCRep, DynCRep, "
        "MTCOV and ACD algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Add the command line arguments

    # Algorithm related arguments
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        choices=["CRep", "JointCRep", "MTCOV", "DynCRep", "ACD"],
        default="CRep",
        help="Choose the algorithm to run: CRep, JointCRep, MTCOV, ACD",
    )
    parser.add_argument(
        "-K", "--K", type=int, default=None, help="Number of communities"
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=0.5,
        help="Scaling hyper parameter in MTCOV",
    )
    parser.add_argument("--rseed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "-nr",
        "--num_realizations",
        type=int,
        default=None,
        help=("Number of realizations"),
    )
    parser.add_argument(
        "-tol",
        "--convergence_tol",
        type=float,
        default=None,
        help=("Tolerance used to determine convergence"),
    )
    parser.add_argument(
        "-T", "--T", type=int, default=None, help="Number of time snapshots"
    )
    parser.add_argument(
        "-fdT",
        "--flag_data_T",
        type=str,
        default=0,
        help="Flag to use data_T. "
        "It is recommended to use 0, but in case it does not work, try 1.",
    )
    parser.add_argument(
        "-no_anomaly",
        "--flag_anomaly",
        action="store_false",
        default=True,
        help="Flag to detect anomalies",
    )
    # TODO: Improve these model specific arguments
    parser.add_argument("-ag", type=float, default=1.1, help="Parameter ag")
    parser.add_argument("-bg", type=float, default=0.5, help="Parameter bg")
    parser.add_argument("-pibr0", default=None, help="Anomaly parameter pi")
    parser.add_argument("-mupr0", default=None, help="Prior mu")
    parser.add_argument(
        "-temp",
        "--temporal",
        action="store_false",
        default=True,
        help="Flag to use non-temporal version of DynCRep. If not set, it will use the temporal "
        "version.",
    )

    # Input/Output related arguments
    parser.add_argument(
        "-A", "--adj_name", type=str, default="syn111.dat", help="Name of the network"
    )
    parser.add_argument(
        "-C",
        "--cov_name",
        type=str,
        default="X.csv",
        help="Name of the design matrix used in MTCOV",
    )
    parser.add_argument(
        "-f", "--in_folder", type=str, default="", help="Path of the input folder"
    )
    parser.add_argument(
        "-o",
        "-O",
        "--out_folder",
        type=str,
        default="",
        help="Path of the output folder",
    )
    parser.add_argument(
        "-out_inference",
        "--out_inference",
        action="store_true",
        default=False,
        help="Flag to save the inference results",
    )

    # Network related arguments
    parser.add_argument(
        "-e", "--ego", type=str, default="source", help="Name of the source of the edge"
    )
    parser.add_argument(
        "-t",
        "--alter",
        type=str,
        default="target",
        help="Name of the target of the edge",
    )
    parser.add_argument(
        "-x",
        "--egoX",
        type=str,
        default="Name",
        help="Name of the column with node labels",
    )
    parser.add_argument(
        "-an",
        "--attr_name",
        type=str,
        default="Metadata",
        help="Name of the attribute to consider in MTCOV",
    )
    parser.add_argument(
        "-u",
        "--undirected",
        type=bool,
        default=False,
        help="Flag to treat the network as undirected",
    )
    parser.add_argument(
        "--noselfloop", type=bool, default=True, help="Flag to remove self-loops"
    )
    parser.add_argument(
        "--binary", type=bool, default=True, help="Flag to make the network binary"
    )

    # Data transformation related arguments
    parser.add_argument(
        "-fd",
        "--force_dense",
        type=bool,
        default=False,
        help="Flag to force a dense transformation in input",
    )
    parser.add_argument(
        "-F",
        "--flag_conv",
        type=str,
        choices=["log", "deltas"],
        default="log",
        help="Flag for convergence",
    )
    parser.add_argument(
        "--plot_loglikelihood",
        type=bool,
        default=False,
        help="Flag to plot the log-likelihood",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=None,
        help="Size of the batch to use to compute the likelihood",
    )

    parser.add_argument(
        "--debug",
        "-d",
        dest="debug",
        action="store_true",
        default=False,
        help="Enable debug mode",
    )

    # Additional parameters
    parser.add_argument(
        "--mask", type=str, default=None, help="Mask for the data"
    ) # TODO: Rethink this. Not sure if the mask can be passed as CLI arg.
    parser.add_argument(
        "--initialization", type=int, default=0, help="Initialization method"
    )
    parser.add_argument(
        "--eta0", type=float, default=None, help="Initial eta value"
    )
    parser.add_argument(
        "--constrained", type=bool, default=False, help="Flag for constrained optimization"
    )
    parser.add_argument(
        "--assortative", type=bool, default=False, help="Flag for assortative mixing"
    )
    parser.add_argument(
        "--use_approximation", type=bool, default=False, help="Flag to use approximation"
    )
    parser.add_argument(
        "--end_file", type=str, default="_CRep", help="Suffix for the output file"
    )
    parser.add_argument(
        "--fix_eta", type=bool, default=False, help="Flag to fix eta"
    )
    parser.add_argument(
        "--fix_w", type=bool, default=False, help="Flag to fix w"
    )
    parser.add_argument(
        "--fix_communities", type=bool, default=False, help="Flag to fix communities"
    )
    parser.add_argument(
        "--files", type=str, default="", help="Path to the input files"
    )
    parser.add_argument(
        "--beta0", type=float, default=0.25, help="Initial beta value"
    )
    parser.add_argument(
        "--constraintU", type=bool, default=False, help="Flag for constraint U"
    )
    parser.add_argument(
        "--fix_beta", type=bool, default=False, help="Flag to fix beta"
    )
    parser.add_argument(
        "--fix_pibr", type=bool, default=False, help="Flag to fix pibr"
    )
    parser.add_argument(
        "--fix_mupr", type=bool, default=False, help="Flag to fix mupr"
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Set the output folder based on the algorithm if not provided
    if args.out_folder == "":
        args.out_folder = args.algorithm + "_output"

    # Map algorithm names to their default adjacency matrix file names
    default_adj_names = {
        "CRep": "syn111.dat",
        "JointCRep": "synthetic_data.dat",
        "MTCOV": "adj.csv",
        "DynCRep": "synthetic_data_for_DynCRep.dat",
        "ACD": "synthetic_data_for_ACD.dat",
    }

    # Correcting default values based on the chosen algorithm
    if args.adj_name == "syn111.dat" and args.algorithm in default_adj_names:
        args.adj_name = default_adj_names[args.algorithm]

    end_file_suffixes = {
        "JointCRep": "_JointCRep",
        "MTCOV": "_MTCOV",
        "DynCRep": "_DynCRep",
        "ACD": "_ACD"
    }

    if args.end_file == "_CRep" and args.algorithm in end_file_suffixes:
        args.end_file = end_file_suffixes[args.algorithm]

    if args.num_realizations is None:
        num_realizations_dict = {"JointCRep": 3, "ACD": 1, "default": 5}
        args.num_realizations = num_realizations_dict.get(
            args.algorithm, num_realizations_dict["default"]
        )

    if args.K is None:
        if args.algorithm in ("MTCOV", "JointCRep", "DynCRep"):
            args.K = 2
        else:
            args.K = 3

    return args


def main():
    """
    Main function for CRep/JointCRep/MTCOV.
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

    # Set the input folder path
    if args.in_folder == "":
        in_folder = (Path(__file__).parent / "data" / "input").resolve()
    else:
        in_folder = args.in_folder
    in_folder = str(in_folder)
    if args.algorithm != "MTCOV":
        if args.algorithm == "DynCRep":
            binary = True  # exactly this in source
            args.force_dense = True  # exactly this in source

        network = in_folder + "/" + args.adj_name
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
            in_folder,
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

        valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
        assert any(isinstance(B, vt) for vt in valid_types)
        logging.debug("Data loaded successfully from %s", in_folder)

    def fit_model(model, algorithm):
        """
        Fit the model to the data.
        """
        if algorithm=='CRep':
            model.fit(data=B,
                      data_T=B_T,
                      data_T_vals=data_T_vals,
                      nodes=nodes,
                      rseed=args.rseed,
                      K=args.K,
                      mask=args.mask,
                      initialization=args.initialization,
                      eta0=args.eta0,
                      undirected=args.undirected,
                      assortative=args.assortative,
                      constrained=args.constrained,
                      out_inference=args.out_inference,
                      out_folder=args.out_folder,
                      end_file=args.end_file,
                      fix_eta=args.fix_eta,
                      files=args.files
                      )
        elif algorithm == "JointCRep":
            model.fit(
                data=B,
                data_T=B_T,
                data_T_vals=data_T_vals,
                nodes=nodes,
                rseed=args.rseed,
                K=args.K,
                initialization=args.initialization,
                eta0=args.eta0,
                undirected=args.undirected,
                assortative=args.assortative,
                use_approximation=args.use_approximation,
                out_inference=args.out_inference,
                out_folder=args.out_folder,
                end_file=args.end_file,
                fix_eta=args.fix_eta,
                fix_w=args.fix_w,
                fix_communities=args.fix_communities,
                files=args.files
            )
        elif algorithm == "DynCRep":
            model.fit(
                data=B,
                T=args.T,
                nodes=nodes,
                mask=args.mask,
                K=args.K,
                rseed=args.rseed,
                ag=args.ag,
                bg=args.bg,
                eta0=args.eta0,
                beta0=args.beta0,
                flag_data_T=args.flag_data_T,
                temporal=args.temporal,
                initialization=args.initialization,
                assortative=args.assortative,
                constrained=args.constrained,
                constraintU=args.constraintU,
                fix_eta=args.fix_eta,
                fix_beta= args.fix_beta,
                fix_communities=args.fix_communities,
                fix_w=args.fix_w,
                undirected=args.undirected,
                out_inference=args.out_inference,
                out_folder=args.out_folder,
                end_file=args.end_file,
                files=args.files,
            )
        elif algorithm == "ACD":
            model.fit(
                data=B,
                nodes=nodes,
                K=args.K,
                undirected=args.undirected,#False,
                initialization=args.initialization,#0,
                rseed=args.rseed,
                assortative=args.assortative,#True,
                constrained=args.constrained,#False,
                ag=args.ag,  # 1.5
                bg=args.bg,  # 10.0
                pibr0=args.pibr0,
                mupr0=args.mupr0,
                flag_anomaly=args.flag_anomaly,
                fix_communities=args.fix_communities,
                fix_pibr=args.fix_pibr,
                fix_mupr=args.fix_mupr,
                out_inference=args.out_inference,
                out_folder=args.out_folder,
                end_file=args.end_file,
                files=args.files,
                verbose=0
            )
        elif algorithm == "MTCOV":
            model.fit(
                data=B,
                data_X=Xs,
                nodes=nodes,
                batch_size=args.batch_size,
                gamma=args.gamma,
                rseed=args.rseed,
                K=args.K,
                initialization=args.initialization,
                undirected=args.undirected,
                assortative=args.assortative,
                out_inference=args.out_inference,
                out_folder=args.out_folder,
                end_file=args.end_file,
                files=args.files,
            )

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
