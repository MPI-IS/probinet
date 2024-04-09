"""
Implementation of the MTCov algorithm.
"""

from argparse import ArgumentParser
from importlib.resources import files
from pathlib import Path
import time

import numpy as np
import sktensor as skt
import yaml

from .input.loader import import_data_mtcov
from .model.mtcov import MTCov


def main():
    '''
    Main function for MTCov.
    '''

    # Step 1: Parse the command-line arguments
    p = ArgumentParser()
    p.add_argument(
        '-f',
        '--in_folder',
        type=str,
        default='')  # path of the input network
    p.add_argument('-j', '--adj_name', type=str, default='adj.csv')  # name of the adjacency tensor
    p.add_argument('-c', '--cov_name', type=str, default='X.csv')  # name of the design matrix
    p.add_argument('-e', '--ego', type=str, default='source',
                   help='Name of the source of the edge')  # name of the source of the edge
    p.add_argument('-t', '--alter', type=str, default='target',
                   help='Name of the target of the edge')  # name of the target of the edge
    p.add_argument('-x', '--egoX', type=str, default='Name')  # name of the column with node labels
    # name of the attribute to consider
    p.add_argument('-a', '--attr_name', type=str, default='Metadata')
    p.add_argument('-K', '--K', type=int, default=2)  # number of communities
    p.add_argument('-g', '--gamma', type=float, default=0.5)  # scaling hyper parameter
    # flag to call the undirected network
    p.add_argument('-u', '--undirected', type=bool, default=False)
    p.add_argument('-F', '--flag_conv', type=str, choices=['log', 'deltas'],
                   default='log')  # flag for convergence
    # flag to force a dense transformation in input
    p.add_argument('-d', '--force_dense', type=bool, default=False)
    # size of the batch used to compute the likelihood
    p.add_argument('-b', '--batch_size', type=int, default=None)
    p.add_argument('-o', '-O', '--out_folder', type=str, default='',
                   help='Path of the output folder')  # path of the output folder
    p.add_argument('-v', '--verbose', action='store_true', help='Print verbose')  # print verbose

    args = p.parse_args()

    in_folder = Path.cwd().resolve() / 'pgm' / 'data' / \
        'input' if args.in_folder == '' else Path(args.in_folder)

    # Step 2: Import the data

    _, B, X, nodes = import_data_mtcov(in_folder,
                                       adj_name=Path(args.adj_name),
                                       cov_name=Path(args.cov_name),
                                       ego=args.ego,
                                       alter=args.alter,
                                       egoX=args.egoX,
                                       attr_name=args.attr_name,
                                       undirected=args.undirected,
                                       force_dense=args.force_dense,
                                       noselfloop=True,
                                       verbose=True)

    Xs = np.array(X)
    valid_types = [np.ndarray, skt.dtensor, skt.sptensor]
    assert any(isinstance(B, vt) for vt in valid_types)

    if args.batch_size and args.batch_size > len(nodes):
        raise ValueError('The batch size has to be smaller than the number of nodes.')
    if len(nodes) > 1000:
        args.flag_conv = 'deltas'

    # Step 3: Load the configuration settings
    with (files('pgm.data.model').joinpath('setting_MTCov.yaml').open('rb')
          as fp):
        conf = yaml.safe_load(fp)

    # Change the output folder
    conf['out_folder'] = 'MTCov' \
        '_output/' if args.out_folder == '' else args.out_folder

    # Change K if given
    if args.K is not None:
        conf['K'] = args.K

    # Step 4: Create the output directory

    out_folder = Path(conf['out_folder'])
    out_folder.mkdir(parents=True, exist_ok=True)

    # Print the configuration file
    if args.verbose:
        print(yaml.dump(conf))

    # Save the configuration file
    output_config_path = conf['out_folder'] + '/setting_' + \
        'MTCov.yaml'
    with open(output_config_path, 'w', encoding='utf8') as f:
        yaml.dump(conf, f)

    # Step 5: Run JointCRep

    if args.verbose:
        print('\n### Run MTCov ###')
    print(f'Setting: \nK = {args.K}\ngamma = {args.gamma}\n')
    print(args)
    time_start = time.time()
    model = MTCov(verbose=args.verbose)
    _ = model.fit(
        data=B,
        data_X=Xs,
        flag_conv=args.flag_conv,
        nodes=nodes,
        batch_size=args.batch_size,
        **conf)
    print("\nTime elapsed:", np.round(time.time() - time_start, 2), " seconds.")


if __name__ == '__main__':
    main()
