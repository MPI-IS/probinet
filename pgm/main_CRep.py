"""
Implementation of CRep algorithm.
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


def main():
    """
    Main function for CRep.
    """
    # Step 1: Parse the command-line arguments
    p = ArgumentParser(description="Script to run the CRep algorithm.",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter
                       )
    # Add the command line arguments
    p.add_argument('-a', '--algorithm', type=str, choices=['CRep'],  # ,# TODO: add this: 'CRepnc', 'CRep0'
                   default='CRep',
                   help='Choose the algorithm to run: CRep, CRepnc, CRep0.')  # configuration
    p.add_argument('-K', '--K', type=int, default=None,
                   help='Number of communities')  # number of communities
    p.add_argument('-A', '--adj', type=str, default='syn111.dat',
                   help='Name of the network')  # name of the network
    p.add_argument('-f', '--in_folder', type=str, default='',
                   help='Path of the input network')  # path of the input network
    p.add_argument('-o', '-O', '--out_folder', type=str, default='',
                   help='Path of the output folder')  # path of the output folder
    p.add_argument('-e', '--ego', type=str, default='source',
                   help='Name of the source of the edge')  # name of the source of the edge
    p.add_argument('-t', '--alter', type=str, default='target',
                   help='Name of the target of the edge')  # name of the target of the edge
    p.add_argument(
        '-d',
        '--force_dense',
        type=bool,
        default=False,
        help='Flag to force a dense transformation in input')
    p.add_argument('-F', '--flag_conv', type=str, choices=['log', 'deltas'], default='log',
                   help='Flag for convergence')
    p.add_argument('-v', '--verbose', action='store_true', help='Print verbose')

    # Parse the command line arguments
    args = p.parse_args()

    # Step 2: Import the data

    ego = args.ego
    alter = args.alter
    force_dense = args.force_dense  # Sparse matrices
    in_folder = Path.cwd().resolve() / 'pgm' / 'data' / \
        'input' if args.in_folder == '' else Path(args.in_folder)
    adj = Path(args.adj)

    network = in_folder / adj  # network complete path
    A, B, B_T, data_T_vals = import_data(
        network,
        ego=ego,
        alter=alter,
        force_dense=force_dense,
        header=0,
        verbose=args.verbose
    )
    nodes = A[0].nodes()

    # Step 3: Load the configuration settings

    config_path = 'setting_' + args.algorithm + '.yaml'
    with files('pgm.data.model').joinpath(config_path).open('rb') as fp:
        conf = yaml.safe_load(fp)

    # Change the output folder
    conf['out_folder'] = args.algorithm + '_output/' if args.out_folder == '' else args.out_folder

    # Change K if given
    if args.K is not None:
        conf['K'] = args.K

    # Print the configuration file
    if args.verbose:
        print(yaml.dump(conf))

    # Step 4: Create the output directory

    out_folder_path = Path(conf['out_folder'])
    out_folder_path.mkdir(parents=True, exist_ok=True)

    # Save the configuration file
    output_config_path = conf['out_folder'] + '/setting_' + \
        args.algorithm + '.yaml'
    with open(output_config_path, 'w', encoding='utf8') as f:
        yaml.dump(conf, f)

    # Run CRep
    # Step 5: Run CRep

    if args.verbose:
        print(f'\n### Run {args.algorithm} ###')
    model = CRep()
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
