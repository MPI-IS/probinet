import unittest
from importlib.resources import files
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from probinet.evaluation.expectation_computation import (
    compute_expected_adjacency_tensor,
)
from probinet.evaluation.link_prediction import compute_link_prediction_AUC
from probinet.input.loader import build_adjacency_from_file
from probinet.models.dyncrep import DynCRep
from probinet.utils.matrix_operations import transpose_tensor
from probinet.utils.tools import flt

from .constants import PATH_FOR_INIT, RANDOM_SEED_REPROD, TOLERANCE_2
from .fixtures import BaseTest


class DynCRepTestCase(BaseTest):
    MAX_ITER = 800
    NUM_REALIZATIONS = 1
    PLOT_LOGLIK = True
    AG = 1.1
    BG = 0.5
    ETA0 = 0.2
    LOG_LIKELIHOOD_EXPECTED = -2872.6923935067616
    PLACES = 3

    def setUp(self):
        self.algorithm = "DynCRep"
        self.label = "GT_DynCRep_for_initialization.npz"
        self.data_path = Path(__file__).parent / "inputs"
        self.theta = np.load(
            (self.data_path / f"theta_{self.label}").with_suffix(".npz"),
            allow_pickle=True,
        )
        self.adj = "dynamic_network.dat"
        self.K = self.theta["u"].shape[1]
        self.gdata = self._import_data()
        self._initialize_graph_properties()

    def _import_data(self, force_dense=True):
        with files("probinet.data.input").joinpath(self.adj).open("rb") as network:
            return build_adjacency_from_file(
                network.name, header=0, force_dense=force_dense
            )

    def _initialize_graph_properties(self):
        self.nodes = self.gdata.nodes
        self.pos = nx.spring_layout(self.gdata.graph_list[0])
        self.N = len(self.nodes)
        self.T = self.gdata.adjacency_tensor.shape[0] - 1

    def _load_and_update_config(self):
        with open(PATH_FOR_INIT / f"setting_{self.algorithm}.yaml") as fp:
            conf = yaml.safe_load(fp)
        conf.update(
            {
                "K": self.K,
                "initialization": 1,
                "out_folder": self.folder,
                "end_file": "_OUT_DynCRep",
                "files": self.data_path / f"theta_{self.label}",
                "constrained": False,
                "undirected": False,
                "eta0": self.ETA0,
                "beta0": self.theta["beta"],
            }
        )
        conf["rng"] = np.random.default_rng(seed=RANDOM_SEED_REPROD)

        return conf

    def assert_model_results_from_yaml(
        self, u, v, w, eta, beta, Loglikelihood, M_inf, B, yaml_file
    ):
        with open(yaml_file, "r") as f:
            expected_values = yaml.safe_load(f)

        self.assertEqual(list(u.shape), expected_values["u"]["shape"])
        self.assertAlmostEqual(
            np.sum(u), expected_values["u"]["sum"], places=self.PLACES
        )
        self.assertEqual(list(v.shape), expected_values["v"]["shape"])
        self.assertAlmostEqual(
            np.sum(v), expected_values["v"]["sum"], places=self.PLACES
        )
        self.assertEqual(list(w.shape), expected_values["w"]["shape"])
        self.assertAlmostEqual(
            np.sum(w), expected_values["w"]["sum"], places=self.PLACES
        )
        self.assertAlmostEqual(eta, expected_values["eta"], places=self.PLACES)
        self.assertAlmostEqual(beta, expected_values["beta"], places=self.PLACES)
        self.assertAlmostEqual(
            Loglikelihood, expected_values["Loglikelihood"], places=self.PLACES
        )

        expected_aucs = expected_values["AUC"]
        for l in range(len(expected_aucs)):
            auc = flt(compute_link_prediction_AUC(M_inf[l], B[l].astype("int")))
            self.assertAlmostEqual(auc, expected_aucs[l], delta=TOLERANCE_2)

    def assert_model_results_from_yaml(
        self, u, v, w, eta, beta, Loglikelihood, M_inf, B, yaml_file
    ):
        with open(yaml_file, "r") as f:
            expected_values = yaml.safe_load(f)

        # Assertions for u
        self.assertEqual(list(u.shape), expected_values["u"]["shape"])
        self.assertAlmostEqual(np.sum(u), expected_values["u"]["sum"], places=3)

        # Assertions for v
        self.assertEqual(list(v.shape), expected_values["v"]["shape"])
        self.assertAlmostEqual(np.sum(v), expected_values["v"]["sum"], places=3)

        # Assertions for w
        self.assertEqual(list(w.shape), expected_values["w"]["shape"])
        self.assertAlmostEqual(np.sum(w), expected_values["w"]["sum"], places=3)

        # Assertions for eta and beta
        self.assertAlmostEqual(eta, expected_values["eta"], places=3)
        self.assertAlmostEqual(beta, expected_values["beta"], places=3)

        # Assertions for Loglikelihood
        self.assertAlmostEqual(
            Loglikelihood, expected_values["Loglikelihood"], places=3
        )

        # Assertions for AUC
        expected_aucs = expected_values["AUC"]
        for l in range(len(expected_aucs)):
            auc = flt(compute_link_prediction_AUC(B[l], M_inf[l]))
            self.assertAlmostEqual(auc, expected_aucs[l], delta=TOLERANCE_2)

    def test_running_temporal_version(self):
        self.conf = self._load_and_update_config()
        model = DynCRep(
            max_iter=self.MAX_ITER,
            num_realizations=self.NUM_REALIZATIONS,
            plot_loglik=self.PLOT_LOGLIK,
        )
        u, v, w, eta, beta, Loglikelihood = model.fit(
            self.gdata,
            T=self.T,
            nodes=self.nodes,
            flag_data_T=0,
            ag=self.AG,
            bg=self.BG,
            **self.conf,
        )

        # Calculate the lambda_inf and M_inf
        lambda_inf = compute_expected_adjacency_tensor(u, v, w[0])
        M_inf = lambda_inf + eta * transpose_tensor(self.gdata.adjacency_tensor)
        yaml_file = (
            Path(__file__).parent
            / "data"
            / "dyncrep"
            / "data_for_test_running_temporal_version.yaml"
        )
        self.assert_model_results_from_yaml(
            u,
            v,
            w,
            eta,
            beta,
            Loglikelihood,
            M_inf,
            self.gdata.adjacency_tensor,
            yaml_file,
        )

        # Load expected values from YAML and assert results
        yaml_file = (
            Path(__file__).parent
            / "data"
            / "dyncrep"
            / "data_for_test_running_temporal_version.yaml"
        )
        self.assert_model_results_from_yaml(
            u,
            v,
            w,
            eta,
            beta,
            Loglikelihood,
            M_inf,
            self.gdata.adjacency_tensor,
            yaml_file,
        )

    @unittest.skip("DynCRep does not support sparse data")
    def test_force_dense_false(self):
        """
        This is a test for the DynCRep algorithm with force_dense=False, i.e., the input data is sparse.
        """
        self.gdata = self._import_data(force_dense=False)
        self._initialize_graph_properties()
        self.conf = self._load_and_update_config()
        model = DynCRep(
            max_iter=self.MAX_ITER,
            num_realizations=self.NUM_REALIZATIONS,
            plot_loglik=self.PLOT_LOGLIK,
        )
        u, v, w, eta, beta, Loglikelihood = model.fit(
            self.gdata,
            T=self.T,
            nodes=self.nodes,
            flag_data_T=0,
            ag=self.AG,
            bg=self.BG,
            **self.conf,
        )
        self.assertEqual(Loglikelihood, self.LOG_LIKELIHOOD_EXPECTED)
