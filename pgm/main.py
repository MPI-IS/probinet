"""
Implementation of CRep and JointCRep algorithm.
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
from .model.crep import CRep
from .model.jointcrep import JointCRep
from .model.mtcov import MTCov


def parse_args():
    """
    Parse the command-line arguments.
    """
    p = argparse.ArgumentParser(description="Script to run the CRep, JointCRep, and MTCov "
                                            "algorithms.",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter
                       )
    # Add the command line arguments

    # Algorithm related arguments
    p.add_argument('-a', '--algorithm', type=str, choices=['CRep', 'JointCRep', 'MTCov'],
                   default='CRep', help='Choose the algorithm to run: CRep, JointCRep.')
    p.add_argument('-K', '--K', type=int, default=None, help='Number of communities')
    p.add_argument('-g', '--gamma', type=float, default=0.5, help='Scaling hyper parameter')
    p.add_argument('--rseed', type=int, default=None, help='Random seed')
    p.add_argument('--num_realizations', type=int, default=None, help='Number of realizations')

    # Input/Output related arguments
    p.add_argument('-A', '--adj_name', type=str, default='syn111.dat', help='Name of the network')
    p.add_argument('-C', '--cov_name', type=str, default='X.csv')
    p.add_argument('-f', '--in_folder', type=str, default='', help='Path of the input folder')
    p.add_argument('-o', '-O', '--out_folder', type=str, default='',
                   help='Path of the output folder')

    # Network related arguments
    p.add_argument('-e', '--ego', type=str, default='source', help='Name of the source of the edge')
    p.add_argument('-t', '--alter', type=str, default='target',
                   help='Name of the target of the edge')
    p.add_argument('-x', '--egoX', type=str, default='Name',
                   help='Name of the column with node labels')
    p.add_argument('-an', '--attr_name', type=str, default='Metadata',
                   help='Name of the attribute to consider')
    p.add_argument('-u', '--undirected', type=bool, default=False,
                   help='Flag to treat the network as undirected')
    p.add_argument('--noselfloop', type=bool, default=True, help='Flag to remove self-loops')
    p.add_argument('--binary', type=bool, default=True, help='Flag to make the network binary')

    # Data transformation related arguments
    p.add_argument('-d', '--force_dense', type=bool, default=False,
                   help='Flag to force a dense transformation in input')
    p.add_argument('-F', '--flag_conv', type=str, choices=['log', 'deltas'], default='log',
                   help='Flag for convergence')
    p.add_argument('--plot_loglikelihood', type=bool, default=False,
                   help='Flag to plot the log-likelihood')
    p.add_argument('-b', '--batch_size', type=int, default=None)

    # Other arguments
    p.add_argument('-l', '--log_level', type=str, choices=['D', 'I', 'W', 'E', 'C'],
                   default='W', help='Set the logging level')
    p.add_argument('--log_file', type=str, default=None, help='Log file to write to')


    # Parse the command line arguments
    args = p.parse_args()

    # Set the output folder based on the algorithm if not provided
    if args.out_folder == '':
        args.out_folder = args.algorithm + '_output'

    # Map algorithm names to their default adjacency matrix file names
    default_adj_names = {
        'CRep': 'syn111.dat',
        'JointCRep': 'synthetic_data.dat',
        'MTCov': 'adj.csv'
    }

    # Correcting default values based on the chosen algorithm
    if args.adj_name == 'syn111.dat' and args.algorithm in default_adj_names:
        args.adj_name = default_adj_names[args.algorithm]

    if args.num_realizations is None:
        if args.algorithm == 'JointCRep':
            args.num_realizations = 3
        else:
            args.num_realizations = 5

    if args.K is None:
        if args.algorithm == 'MTCov':
            args.K = 2
        elif args.algorithm == 'JointCRep':
            args.K = 4
        else:
            args.K = 3

    return args


def main(): # pylint: disable=too-many-branches, too-many-statements
    """
    Main function for CRep/JointCRep/MTCov.
    """
    # Step 1: Parse the command-line arguments

    args = parse_args()

    # Configure the logger
    configure_logger(args)

    # Step 2: Import the data

    # Set default values
    binary = True
    noselfloop = True
    undirected = False

    # Change values if the algorithm is not 'CRep'
    if args.algorithm != 'CRep':
        binary = args.binary
        noselfloop = args.noselfloop
        undirected = args.undirected

    # Set the input folder path
    if args.in_folder == '':
        in_folder = (Path(__file__).parent / 'data' / 'input').resolve()
    else:
        in_folder = args.in_folder
    in_folder = str(in_folder)
    if args.algorithm != 'MTCov':
        network = in_folder + '/' + args.adj_name
        A, B, B_T, data_T_vals = import_data(network,
                                             ego=args.ego,
                                             alter=args.alter,
                                             undirected=undirected,
                                             force_dense=args.force_dense,
                                             noselfloop=noselfloop,
                                             binary=binary,
                                             header=0)
        nodes = A[0].nodes()
        Xs = None
    else:
        A, B, X, nodes = import_data_mtcov(in_folder,
                                           adj_name=args.adj_name,
                                           cov_name=args.cov_name,
                                           ego=args.ego,
                                           alter=args.alter,
                                           egoX=args.egoX,
                                           attr_name=args.attr_name,
                                           undirected=undirected,
                                           force_dense=args.force_dense,
                                           noselfloop=True
                                           )
        Xs = np.array(X)
        B_T = None
        data_T_vals = None

        valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
        assert any(isinstance(B, vt) for vt in valid_types)

    # Step 3: Load the configuration settings

    config_path = 'setting_' + args.algorithm + '.yaml'
    with files('pgm.data.model').joinpath(config_path).open('rb') as fp:
        conf = yaml.safe_load(fp)

    # Change the output folder
    conf['out_folder'] = args.out_folder

    def set_config(args, conf):
        """
        Set the configuration file based on the command line arguments.
        """
        # Change K if given
        if args.K is not None:
            conf['K'] = args.K
        if args.rseed is not None: # if it has a value, then update the configuration
            conf['rseed'] = args.rseed

        # Algorithm specific settings
        algorithm_settings = {
            'JointCRep': {
                'plot_loglik': args.plot_loglikelihood,
                'num_realizations': args.num_realizations
            },
            'MTCov': {
                'gamma': args.gamma
            }
        }

        if args.algorithm in algorithm_settings:
            conf.update(algorithm_settings[args.algorithm])

        return conf

    # Use the function to set the configuration. We need to update the config
    # file based on the command line arguments.
    conf = set_config(args, conf)

    # Print the configuration file
    logging.info('\n' + yaml.dump(conf))

    # Step 4: Create the output directory

    # Create the output directory
    out_folder_path = Path(conf['out_folder'])
    out_folder_path.mkdir(parents=True, exist_ok=True)

    # Save the configuration file
    output_config_path = conf['out_folder'] + '/setting_' + args.algorithm + '.yaml'
    with open(output_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(conf, f)

    def fit_model(model, algorithm, B, B_T, data_T_vals, Xs, nodes, conf): # pylint: disable=too-many-arguments
        """
        Fit the model to the data.
        """
        if algorithm != 'MTCov':
            model.fit(data=B, data_T=B_T, data_T_vals=data_T_vals, nodes=nodes, **conf)
        else:
            model.fit(data=B, data_X=Xs, flag_conv=args.flag_conv, nodes=nodes,
                      batch_size=args.batch_size, **conf)

    # Step 5: Run the algorithm

    logging.info(f'### Running {args.algorithm} ###')
    logging.info(f'Setting: K = {conf["K"]}')
    if args.algorithm == 'MTCov':
        logging.info(f'gamma = {conf["gamma"]}')

    # Map algorithm names to their classes
    algorithm_classes = {'CRep': CRep, 'JointCRep': JointCRep, 'MTCov': MTCov}

    # Create the model
    if args.algorithm in algorithm_classes:
        model = algorithm_classes[args.algorithm](
            flag_conv=args.flag_conv)

    else:
        message = 'Algorithm not implemented.'
        error_type = ValueError
        log_and_raise_error(logging, error_type, message)

    # Time the execution
    time_start = time.time()

    # Fit the model to the data
    fit_model(model, args.algorithm, B, B_T, data_T_vals, Xs, nodes, conf)

    # Print the time elapsed
    logging.info(f'Time elapsed: {np.round(time.time() - time_start, 2)} seconds.')


def configure_logger(args):
    '''
    Configure the logger based on the command-line arguments.
    '''

    # Map the first letter to the corresponding logging level
    log_level_map = {
        'D': 'DEBUG',
        'I': 'INFO',
        'W': 'WARNING',
        'E': 'ERROR',
        'C': 'CRITICAL'
    }
    # Set the logging level based on the first letter
    log_level = log_level_map.get(args.log_level.upper(), 'WARNING')

    # Configure the logger

    # Get the numeric value of the log level
    numeric_level = getattr(logging, log_level, None)

    # Check if the log level is valid
    if not isinstance(numeric_level, int):
        # If the log level is not valid, raise an error
        raise ValueError(f'Invalid log level: {args.log_level}')

    # Check if a log file was provided
    if args.log_file is not None:
        # If a log file was provided, configure the logger to write to that file
        logging.basicConfig(filename=args.log_file, level=numeric_level,
                            format='%(levelname)s:%(message)s')
    else:
        # If no log file was provided, configure the logger to write to the console
        logging.basicConfig(level=numeric_level, format='%(levelname)s:%(message)s')

    # Set the log level for the root logger
    logging.getLogger().setLevel(numeric_level)

    # Log the current date and time
    logging.info(f"Program started at {time.strftime('%Y-%m-%d %H:%M:%S')}")