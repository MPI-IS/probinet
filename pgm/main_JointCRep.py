"""
Performing the inference in the given single-layer directed network.
Implementation of JointCRep algorithm.
"""

from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path
import time

import numpy as np
import yaml

from .input.loader import import_data
from .model.jointcrep import JointCRep

# pylint: disable=too-many-locals


def main():
    """
    Main function for JointCRep.
    """
    p = ArgumentParser()
    p.add_argument('-a', '--algorithm', type=str, default='JointCRep')
    p.add_argument('-K', '--K', type=int, default=4)  # number of communities
    p.add_argument('-A', '--adj', type=str, default='highschool_data.dat')
    p.add_argument('-f', '--in_folder', type=str, default='')  # path of the input folder
    p.add_argument('-o', '-O', '--out_folder', type=str, default='')  # path of the output folder
    p.add_argument('-e', '--ego', type=str, default='source')  # name of the source of the edge
    p.add_argument('-t', '--alter', type=str, default='target')  # name of the target of the edge
    # flag to call the undirected network
    p.add_argument('--undirected', type=bool, default=False)
    # flag to force a dense transformation in input
    p.add_argument('-d', '--force_dense', type=bool, default=False)
    p.add_argument(
        '-F',
        '--flag_conv',
        type=str,
        choices=['log', 'deltas'],
        default='log')  # flag for convergence
    p.add_argument('--noselfloop', type=bool, default=True)  # flag to remove self-loops
    p.add_argument('--binary', type=bool, default=True)   # flag to make the network binary
    # flag to plot the log-likelihood
    p.add_argument('--plot_loglikelihood', type=bool, default=False)
    p.add_argument('--rseed', type=int, default=0)  # random seed
    p.add_argument('--num_realizations', type=int, default=50)  # number of realizations
    p.add_argument('-v', '--verbose', action='store_true')  # print verbose
    args = p.parse_args()

    # setting to run the algorithm

    config_path = 'setting_' + args.algorithm + '.yaml'
    with files('pgm.data.model').joinpath(config_path).open('rb') as fp:
        conf = yaml.safe_load(fp)

    # Change the output folder
    conf['out_folder'] = args.algorithm + \
        '_output/' if args.out_folder == '' else args.out_folder

    # Change K if given
    if args.K is not None:
        conf['K'] = args.K

    # Assign argparse arguments to other constants
    conf['plot_loglik'] = args.plot_loglikelihood
    conf['rseed'] = args.rseed
    conf['num_realizations'] = args.num_realizations

    # Ensure the output folder exists
    out_folder_path = Path(conf['out_folder'])
    out_folder_path.mkdir(parents=True, exist_ok=True)

    # Print the configuration file
    if args.verbose:
        print(yaml.dump(conf))

    # Save the configuration file
    output_config_path = conf['out_folder'] + '/setting_' + args.algorithm + '.yaml'
    with open(output_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(conf, f)

    # Import data
    ego = args.ego
    alter = args.alter
    force_dense = args.force_dense  # Sparse matrices
    in_folder = Path.cwd().resolve() / 'pgm' / 'data' / \
        'input' if args.in_folder == '' else Path(args.in_folder)
    adj = Path(args.adj)

    # Import data: removing self-loops and making binary
    network = in_folder / adj  # network complete path
    A, B, B_T, data_T_vals = import_data(network,
                                         ego=ego,
                                         alter=alter,
                                         undirected=args.undirected,
                                         force_dense=force_dense,
                                         noselfloop=args.noselfloop,
                                         verbose=args.verbose,
                                         binary=args.binary,
                                         header=0)
    nodes = A[0].nodes()

    # Run model
    if args.verbose:
        print(f'\n### Run {args.algorithm} ###')
    model = JointCRep()
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
