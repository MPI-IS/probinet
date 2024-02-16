"""
Implementation of CRep and JointCRep algorithm.
"""

import argparse
from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path
import time

import numpy as np
import sktensor as skt
import yaml

from .input.loader import import_data, import_data_mtcov
from .model.crep import CRep
from .model.jointcrep import JointCRep
from .model.mtcov import MTCov


def parse_args():
    """
    Parse the command-line arguments.
    """
    p = ArgumentParser(description="Script to run the CRep, JointCRep, and MTCov algorithms.",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter
                       )
    # Add the command line arguments

    # Algorithm related arguments
    p.add_argument('-a', '--algorithm', type=str, choices=['CRep', 'JointCRep', 'MTCov'],
                   default='CRep', help='Choose the algorithm to run: CRep, JointCRep.')
    p.add_argument('-K', '--K', type=int, default=None, help='Number of communities')
    p.add_argument('-g', '--gamma', type=float, default=0.5, help='Scaling hyper parameter')
    p.add_argument('--rseed', type=int, default=None, help='Random seed')
    p.add_argument('--num_realizations', type=int, default=5, help='Number of realizations')

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
    p.add_argument('-v', '--verbose', action='store_true', help='Print verbose')

    # Parse the command line arguments
    return p.parse_args()


def main():
    """
    Main function for CRep/JointCRep.
    """
    # Step 1: Parse the command-line arguments

    args = parse_args()

    # Map algorithm names to their default adjacency matrix file names
    default_adj_names = {
        'CRep': 'syn111.dat',
        'JointCRep': 'highschool_data.dat',
        'MTCov': 'adj.csv'
    }

    # Correcting default values based on the chosen algorithm
    if args.adj_name == 'syn111.dat' and args.algorithm in default_adj_names:
        args.adj_name = default_adj_names[args.algorithm]

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
    in_folder = Path.cwd().resolve() / 'pgm' / 'data' / \
        'input' if args.in_folder == '' else Path(args.in_folder)

    if args.algorithm != 'MTCov':
        network = in_folder / args.adj_name
        A, B, B_T, data_T_vals = import_data(network,
                                             ego=args.ego,
                                             alter=args.alter,
                                             undirected=undirected,
                                             force_dense=args.force_dense,
                                             noselfloop=noselfloop,
                                             verbose=args.verbose,
                                             binary=binary,
                                             header=0)
        nodes = A[0].nodes()
        Xs = None
    else:
        A, B, X, nodes = import_data_mtcov(in_folder,
                                           adj_name=Path(args.adj_name),
                                           cov_name=Path(args.cov_name),
                                           ego=args.ego,
                                           alter=args.alter,
                                           egoX=args.egoX,
                                           attr_name=args.attr_name,
                                           undirected=undirected,
                                           force_dense=args.force_dense,
                                           noselfloop=True,
                                           verbose=args.verbose)
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
    conf['out_folder'] = args.algorithm + \
        '_output/' if args.out_folder == '' else args.out_folder

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
    if args.verbose:
        print(yaml.dump(conf))

    # Step 4: Create the output directory

    # Create the output directory
    out_folder_path = Path(conf['out_folder'])
    out_folder_path.mkdir(parents=True, exist_ok=True)

    # Save the configuration file
    output_config_path = conf['out_folder'] + '/setting_' + args.algorithm + '.yaml'
    with open(output_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(conf, f)

    def fit_model(model, algorithm, B, B_T, data_T_vals, Xs, nodes, conf):
        """
        Fit the model to the data.
        """
        if algorithm != 'MTCov':
            model.fit(data=B, data_T=B_T, data_T_vals=data_T_vals, nodes=nodes, **conf)
        else:
            model.fit(data=B, data_X=Xs, flag_conv=args.flag_conv, nodes=nodes,
                      batch_size=args.batch_size, **conf)

    # Step 5: Run the algorithm

    # Print verbose information if verbose flag is set
    if args.verbose:
        print(f'\n### Running {args.algorithm} ###')
        print(f'Setting: \nK = {conf["K"]}')
        if args.algorithm == 'MTCov':
            print(f'gamma = {conf["gamma"]}')

    # Map algorithm names to their classes
    algorithm_classes = {'CRep': CRep, 'JointCRep': JointCRep, 'MTCov': MTCov}

    # Create the model
    if args.algorithm in algorithm_classes:
        model = algorithm_classes[args.algorithm](verbose=args.verbose, flag_conv=args.flag_conv,
                                                  num_realizations=args.num_realizations)
    else:
        raise ValueError('Algorithm not implemented.')

    # Time the execution
    time_start = time.time()

    # Fit the model to the data
    fit_model(model, args.algorithm, B, B_T, data_T_vals, Xs, nodes, conf)

    # Print the time elapsed
    if args.verbose:
        print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')


if __name__ == '__main__':
    main()
