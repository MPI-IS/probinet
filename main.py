"""
    Performing the inference in the given single-layer directed network.
    Implementation of CRep algorithm.
"""

import os
import time
from argparse import ArgumentParser

import numpy as np
import sktensor as skt
import yaml

from src.pgm.input.loader import import_data
from src.pgm.model import CRep as CREP


def main():
    p = ArgumentParser()
    p.add_argument('-a',
                   '--algorithm',
                   type=str,
                   choices=['Crep', 'Crepnc', 'Crep0'],
                   default='CRep')  # configuration
    p.add_argument('-K', '--K', type=int, default=3)  # number of communities
    p.add_argument('-A', '--adj', type=str,
                   default='syn111.dat')  # name of the network
    p.add_argument('-f',
                   '--in_folder',
                   type=str,
                   default='src/pgm/data/input/')  # path of the input network
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
    with open('src/pgm/input/setting_' + args.algorithm + '.yaml') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    if not os.path.exists(conf['out_folder']):
        os.makedirs(conf['out_folder'])
    with open(conf['out_folder'] + '/setting_' + args.algorithm + '.yaml',
              'w') as f:
        yaml.dump(conf, f)
    '''
    Import data
    '''
    network = args.in_folder + args.adj  # network complete path
    A, B, B_T, data_T_vals = import_data(network,
                                         ego=args.ego,
                                         alter=args.alter,
                                         force_dense=args.force_dense,
                                         header=0)
    nodes = A[0].nodes()

    valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
    assert any(isinstance(B, vt) for vt in valid_types)
    '''
    Run CRep    
    '''
    print(f'\n### Run {args.algorithm} ###')

    time_start = time.time()
    model = CREP.CRep(N=A[0].number_of_nodes(), L=len(A), K=args.K, **conf)
    _ = model.fit(data=B,
                  data_T=B_T,
                  data_T_vals=data_T_vals,
                  flag_conv=args.flag_conv,
                  nodes=nodes)

    print(f'\nTime elapsed: {np.round(time.time() - time_start, 2)} seconds.')


if __name__ == '__main__':
    main()
