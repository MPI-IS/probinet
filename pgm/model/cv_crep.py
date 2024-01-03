"""
It provides functions for cross-validation for the CRep model.
"""
import yaml

from pgm.model.crep import CRep

# pylint: disable=too-many-arguments, too-many-instance-attributes, too-many-locals, too-many-branches, too-many-statements
# pylint: disable=fixme

def fit_model(B, B_T, data_T_vals, nodes, algo, **conf):
    """
    Model directed networks by using a probabilistic generative model that assume community
    parameters and reciprocity coefficient. The inference is performed via EM algorithm.

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
    model : obj
          The CRep object.
    """

    # setting to run the algorithm
    with open(conf['out_folder'] + '/setting_' + algo + '.yaml', 'w', encoding='utf8') as f:
        yaml.dump(conf, f)

    model = CRep()
    uf, vf, wf, nuf, maxPSL = model.fit(data=B,
                                        data_T=B_T,
                                        data_T_vals=data_T_vals,
                                        nodes=nodes,
                                        **conf)

    return uf, vf, wf, nuf, maxPSL, model
