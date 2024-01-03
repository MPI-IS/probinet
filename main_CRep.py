"""
Performing the inference in the given single-layer directed network.
Implementation of CRep algorithm.
"""

from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path
import time

import numpy as np
import yaml

from pgm.input.loader import import_data
from pgm.model.crep import CRep

# pylint: disable=too-many-locals


def main():
    """
    Main function for CRep.
    """
    p = ArgumentParser()
    p.add_argument('-a',
                   '--algorithm',
                   type=str,
                   choices=['Crep', 'Crepnc', 'Crep0'],
                   default='CRep')  # configuration
    p.add_argument('-K', '--K', type=int, default=None)  # number of communities
    p.add_argument('-A', '--adj', type=str,
                   default='syn111.dat')  # name of the network
    p.add_argument('-f',
                   '--in_folder',
                   type=str,
                   default='')  # path of the input network
    p.add_argument('-o', '-O', '--out_folder', type=str, default='')  # path of the output folder
    p.add_argument('-e', '--ego', type=str,
                   default='source')  # name of the source of the edge
    p.add_argument('-t', '--alter', type=str,
                   default='target')  # name of the target of the edge
    p.add_argument(
        '-d', '--force_dense', type=bool,
        default=False)  # flag to force a dense transformation in input
    p.add_argument('-F',
                   '--flag_conv',
                   type=str,
                   choices=['log', 'deltas'],
                   default='log')  # flag for convergence

    args = p.parse_args()

    # setting to run the algorithm

    config_path = 'setting_' + args.algorithm + '.yaml'
    with files('pgm.data.model').joinpath(config_path).open('rb') as fp:
        conf = yaml.safe_load(fp)

    # Change the output folder
    conf['out_folder'] = './' + \
                         args.algorithm + '_output/' if args.out_folder == '' else args.out_folder

    # Change K if given
    if args.K is not None:
        conf['K'] = args.K

    # Ensure the output folder exists
    out_folder_path = Path(conf['out_folder'])

    if not out_folder_path.exists():
        out_folder_path.mkdir(parents=True)

    # Print the configuration file
    print(yaml.dump(conf))

    # Save the configuration file
    output_config_path = conf['out_folder'] + '/setting_' + args.algorithm + '.yaml'
    with open(output_config_path, 'w', encoding='utf8') as f:
        yaml.dump(conf, f)

    # Import data
    ego = args.ego
    alter = args.alter
    force_dense = args.force_dense  # Sparse matrices
    in_folder = Path.cwd().resolve() / 'pgm' / 'data' / \
        'input' if args.in_folder == '' else Path(args.in_folder)
    adj = Path(args.adj)
    network = in_folder / adj  # network complete path

    A, B, B_T, data_T_vals = import_data(network,
                                         ego=ego,
                                         alter=alter,
                                         force_dense=force_dense,
                                         header=0)
    nodes = A[0].nodes()

    # Run CRep

    print(f'\n### Run {args.algorithm} ###')
    model = CRep()
    time_start = time.time()
    _ = model.fit(data=B,
                  data_T=B_T,
                  data_T_vals=data_T_vals,
                  nodes=nodes,
                  **conf)

    print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')


if __name__ == '__main__':
    main()
