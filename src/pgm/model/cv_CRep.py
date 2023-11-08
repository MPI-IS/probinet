"""
It provides functions for cross-validation for the CRep model.
"""

import yaml

from . import CRep as CREP


def fit_model(B, B_T, data_T_vals, nodes, N, L, algo, K, flag_conv, **conf):
    """
    Model directed networks by using a probabilistic generative model that assume community parameters and
    reciprocity coefficient. The inference is performed via EM algorithm.

    Parameters
    ----------
    B : ndarray
        Graph adjacency tensor.
    B_T : None/sptensor
          Graph adjacency tensor (transpose).
    data_T_vals : None/ndarray
                  Array with values of entries A[j, i] given non-zero entry (i, j).
    nodes : list
            List of nodes IDs.
    N : int
        Number of nodes.
    L : int
        Number of layers.
    algo : str
           Configuration to use (CRep, CRepnc, CRep0).
    K : int
        Number of communities.
    flag_conv : str
                If 'log' the convergence is based on the log-likelihood values; if 'deltas' the convergence is
                based on the differences in the parameters values. The latter is suggested when the dataset
                is big (N > 1000 ca.).

    Returns
    -------
    u_f : ndarray
          Out-going membership matrix.
    v_f : ndarray
          In-coming membership matrix.
    w_f : ndarray
          Affinity tensor.
    eta_f : float
            Reciprocity coefficient.
    maxPSL : float
             Maximum pseudo log-likelihood.
    mod : obj
          The CRep object.
    """

    # setting to run the algorithm
    with open(conf['out_folder'] + '/setting_' + algo + '.yaml', 'w') as f:
        yaml.dump(conf, f)

    mod = CREP.CRep(N=N, L=L, K=K, **conf)
    uf, vf, wf, nuf, maxPSL = mod.fit(data=B,
                                      data_T=B_T,
                                      data_T_vals=data_T_vals,
                                      flag_conv=flag_conv,
                                      nodes=nodes)

    return uf, vf, wf, nuf, maxPSL, mod
