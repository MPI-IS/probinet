"""
Performing the inference in the given single-layer directed network.
Implementation of JointCRep algorithm.
"""

import importlib.resources as importlib_resources
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import yaml

from pgm.input.loader import import_data
from pgm.model.jointcrep import JointCRep


def main():
    p = ArgumentParser()
    p.add_argument('-a',
                   '--algorithm',
                   type=str,
                   default='JointCRep')
    p.add_argument('-K', '--K', type=int, default=4)  # number of communities
    p.add_argument('-A', '--adj', type=str, default='highschool_data.dat')
    p.add_argument(
        '-f',
        '--in_folder',
        type=str,
        default='')  # path of the input folder
    p.add_argument('-o', '-O', '--out_folder', type=str, default='')  # path of the output folder
    p.add_argument('-e', '--ego', type=str,
                   default='source')  # name of the source of the edge
    p.add_argument('-t', '--alter', type=str,
                   default='target')  # name of the target of the edge
    # flag to call the undirected network
    p.add_argument('-u', '--undirected', type=bool, default=False)
    # flag to force a dense transformation in input
    p.add_argument('-d', '--force_dense', type=bool, default=False)
    p.add_argument(
        '-F',
        '--flag_conv',
        type=str,
        choices=[
            'log',
            'deltas'],
        default='log')  # flag for convergence

    args = p.parse_args()

    # setting to run the algorithm

    config_path = 'setting_' + args.algorithm + '.yaml'
    with importlib_resources.open_binary('pgm.data.model', config_path) as fp:
        conf = yaml.load(fp, Loader=yaml.Loader)

    # Change the output folder
    conf['out_folder'] = './' + args.algorithm + \
                         '_output/' if args.out_folder == '' else args.out_folder
    conf['plot_loglik'] = False
    conf['rseed'] = 0
    conf['N_real'] = 50
    # Ensure the output folder exists
    if not os.path.exists(conf['out_folder']):
        os.makedirs(conf['out_folder'])

    # Print the configuration file
    print(yaml.dump(conf))

    # Save the configuration file
    output_config_path = conf['out_folder'] + '/setting_' + args.algorithm + '.yaml'
    with open(output_config_path, 'w') as f:
        yaml.dump(conf, f)

    # Import data
    ego = args.ego
    alter = args.alter
    force_dense = args.force_dense  # Sparse matrices
    in_folder = Path.cwd().resolve() / 'pgm/data/input/' if args.in_folder == '' else Path(args.in_folder)
    adj = Path(args.adj)

    undirected = False
    noselfloop = True
    verbose = True
    binary = True

    '''
    Import data: removing self-loops and making binary
    '''
    network = in_folder / adj  # network complete path
    A, B, B_T, data_T_vals = import_data(network,
                                         ego=ego,
                                         alter=alter,
                                         undirected=undirected,
                                         force_dense=force_dense,
                                         noselfloop=noselfloop,
                                         verbose=verbose,
                                         binary=binary,
                                         header=0)
    nodes = A[0].nodes()
    N = len(nodes)
    L = len(A)
    undirected = False
    flag_conv = 'log'
    '''
    Run model
    '''
    # TODO: Refactor the code to avoid the following line (done in a CSD-269 for CRep)
    model = JointCRep(N=N, L=L, K=args.K, undirected=undirected, **conf)
    time_start = time.time()
    _ = model.fit(data=B,
                  data_T=B_T,
                  data_T_vals=data_T_vals,
                  flag_conv=flag_conv,
                  nodes=nodes)

    print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')


if __name__ == '__main__':
    main()
