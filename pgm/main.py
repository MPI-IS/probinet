"""
Implementation of CRep and JointCRep algorithm.
"""

import argparse
from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path
import time

import numpy as np
import yaml

from .input.loader import import_data
from .model.crep import CRep
from .model.jointcrep import JointCRep


def main():
    """
    Main function for CRep/JointCRep.
    """
    # Step 1: Parse the command-line arguments
    p = ArgumentParser(description="Script to run the CRep and JointCrep algorithm.",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter
                       )
    # Add the command line arguments
    p.add_argument('-a', '--algorithm', type=str, choices=['CRep', 'JointCRep'],  # ,# TODO: add
                   # this:
                   # 'CRepnc', 'CRep0'
                   default='CRep',
                   help='Choose the algorithm to run: CRep, JointCRep.')  # configuration
    p.add_argument('-K', '--K', type=int, default=None,
                   help='Number of communities')
    p.add_argument('-A', '--adj', type=str, default='syn111.dat',
                   help='Name of the network') # default for JointCRep: highschool_data.dat
    p.add_argument('-f', '--in_folder', type=str, default='',
                   help='Path of the input folder')
    p.add_argument('-o', '-O', '--out_folder', type=str, default='',
                   help='Path of the output folder')
    p.add_argument('-e', '--ego', type=str, default='source',
                   help='Name of the source of the edge')
    p.add_argument('-t', '--alter', type=str, default='target',
                   help='Name of the target of the edge')
    p.add_argument(
        '-d',
        '--force_dense',
        type=bool,
        default=False,
        help='Flag to force a dense transformation in input')

    p.add_argument('--undirected', type=bool, default=False,
                   help='Flag to treat the network as undirected')
    p.add_argument('-F', '--flag_conv', type=str, choices=['log', 'deltas'], default='log',
                   help='Flag for convergence')
    p.add_argument('--noselfloop', type=bool, default=True,
                   help='Flag to remove self-loops')
    p.add_argument('--binary', type=bool, default=True,
                   help='Flag to make the network binary')
    p.add_argument(
        '--plot_loglikelihood',
        type=bool,
        default=False,
        help='Flag to plot the log-likelihood')
    p.add_argument('--rseed', type=int, default=0, help='Random seed')
    p.add_argument('--num_realizations', type=int, default=50,
                   help='Number of realizations')

    p.add_argument('-v', '--verbose', action='store_true', help='Print verbose')

    # Parse the command line arguments
    args = p.parse_args()

    # Correcting defaults

    if args.algorithm == 'CRep':
        if args.adj == 'syn111.dat':  # The default value for CRep
            args.adj = 'syn111.dat'
    elif args.algorithm == 'JointCRep':
        if args.adj == 'syn111.dat':  # The default value for CRep
            args.adj = 'highschool_data.dat'  # The default value for JointCRep

    if args.verbose:
        print(f'\n### Running {args.algorithm} ###')

    # Step 2: Import the data

    # Only in JointCRep
    if args.algorithm == 'JointCRep':
        binary = args.binary
        noselfloop = args.noselfloop
        undirected = args.undirected
    else:
        binary = True
        noselfloop = True
        undirected = False

    in_folder = Path.cwd().resolve() / 'pgm' / 'data' / \
        'input' if args.in_folder == '' else Path(args.in_folder)
    adj = Path(args.adj)

    # Import data: removing self-loops and making binary
    network = in_folder / adj  # network complete path

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

    # Step 3: Load the configuration settings

    config_path = 'setting_' + args.algorithm + '.yaml'
    with files('pgm.data.model').joinpath(config_path).open('rb') as fp:
        conf = yaml.safe_load(fp)

    # Change the output folder
    conf['out_folder'] = args.algorithm + \
        '_output/' if args.out_folder == '' else args.out_folder


    # Change K if given
    if args.K is not None:
        conf['K'] = args.K

    if args.algorithm == 'JointCRep':
        conf['plot_loglik'] = args.plot_loglikelihood
        conf['rseed'] = args.rseed
        conf['num_realizations'] = args.num_realizations

    # Print the configuration file
    if args.verbose:
        print(yaml.dump(conf))

    # Step 4: Create the output directory
    out_folder_path = Path(conf['out_folder'])
    out_folder_path.mkdir(parents=True, exist_ok=True)

    output_config_path = conf['out_folder'] + '/setting_' + args.algorithm + '.yaml'
    with open(output_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(conf, f)

    # Step 5: Run the algorithm

    if args.algorithm == 'CRep':
        model = CRep(verbose=args.verbose)
    elif args.algorithm == 'JointCRep':
        model = JointCRep(verbose=args.verbose)
    else:
        raise ValueError('Algorithm not implemented.')

    time_start = time.time()
    _ = model.fit(data=B,
                  data_T=B_T,
                  data_T_vals=data_T_vals,
                  nodes=nodes,
                  **conf)
    if args.verbose:
        print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')


if __name__ == '__main__':
    main()
