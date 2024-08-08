"""
Implementation of CRep, JointCRep, and MTCOV algorithm.
"""

import argparse
from importlib.resources import files
import logging
from pathlib import Path
import time

import numpy as np
import sktensor as skt
import yaml

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
    parser.add_argument("--rseed", type=int, default=None, help="Random seed")
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


def main():  # pylint: disable=too-many-branches, too-many-statements
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
    # Step 3: Load the configuration settings
    if args.algorithm == "ACD":
        logging.debug("Building configuration file for ACD.")
        conf = {
            "K": args.K,
            "fix_communities": False,
            "files": args.in_folder,
            "out_inference": args.out_inference,
            "end_file": "_ACD",
            # "end_file": self.label,
            "verbose": 1,
        }
    else:
        logging.debug("Loading the configuration file: setting_%s.yaml", args.algorithm)
        config_path = "setting_" + args.algorithm + ".yaml"
        with files("pgm.data.model").joinpath(config_path).open("rb") as fp:
            conf = yaml.safe_load(fp)

    def set_config(args, conf):
        """
        Set the configuration file based on the command line arguments.
        """
        # Change K if given
        if args.K is not None:
            conf["K"] = args.K
        if args.rseed is not None:  # if it has a value, then update the configuration
            conf["rseed"] = args.rseed
        # Change the output inference flag
        conf["out_inference"] = args.out_inference
        # Change the output folder
        conf["out_folder"] = args.out_folder

        # Algorithm specific settings
        algorithm_settings = {"MTCOV": {"gamma": args.gamma}}

        if args.algorithm in algorithm_settings:
            conf.update(algorithm_settings[args.algorithm])

        return conf

    # Use the function to set the configuration. We need to update the config
    # file based on the command line arguments.
    conf = set_config(args, conf)

    # Print the configuration file
    logging.debug("The configuration file is: %s", yaml.dump(conf))

    # Step 4 (Optional): Create the output directory and store the configuration file
    if conf["out_inference"]:

        # Create the output directory
        out_folder_path = Path(conf["out_folder"])
        out_folder_path.mkdir(parents=True, exist_ok=True)

        # Save the configuration file
        output_config_path = conf["out_folder"] + "/setting_" + args.algorithm + ".yaml"
        with open(output_config_path, "w", encoding="utf-8") as f:
            yaml.dump(conf, f)

    def fit_model(model, algorithm, conf):
        """
        Fit the model to the data.
        """
        if algorithm in {"CRep", "JointCRep"}:
            model.fit(data=B, data_T=B_T, data_T_vals=data_T_vals, nodes=nodes, **conf)
        elif algorithm == "DynCRep":
            model.fit(
                data=B,
                T=args.T,
                nodes=nodes,
                flag_data_T=args.flag_data_T,
                ag=args.ag,
                bg=args.bg,
                temporal=args.temporal,
                **conf,
            )
        elif algorithm == "ACD":
            model.fit(
                data=B,
                nodes=nodes,
                undirected=False,
                initialization=0,
                assortative=True,
                constrained=False,
                ag=args.ag,  # 1.5
                bg=args.bg,  # 10.0
                pibr0=args.pibr0,
                mupr0=args.mupr0,
                flag_anomaly=args.flag_anomaly,
                fix_pibr=False,
                fix_mupr=False,
                **conf,
            )
        else:
            model.fit(
                data=B,
                data_X=Xs,
                flag_conv=args.flag_conv,
                nodes=nodes,
                batch_size=args.batch_size,
                **conf,
            )

    # Step 5: Run the algorithm

    logging.info("Setting: K = %s", conf["K"])
    if args.algorithm == "MTCOV":
        logging.info("gamma = %s", conf["gamma"])
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
    fit_model(model, args.algorithm, conf)

    # Print the time elapsed
    logging.info("Time elapsed: %.2f seconds.", time.time() - time_start)
