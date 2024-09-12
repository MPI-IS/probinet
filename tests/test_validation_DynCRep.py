from importlib.resources import files
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from pgm.input.loader import import_data
from pgm.input.tools import flt, transpose_tensor
from pgm.model.dyncrep import DynCRep
from pgm.output.evaluate import calculate_AUC, expected_Aija

from .constants import PATH_FOR_INIT, TOLERANCE_2
from .fixtures import BaseTest


class DynCRepTestCase(BaseTest):
    def setUp(self):
        # Test case parameters
        self.algorithm = "DynCRep"
        self.label = (
            "GT_DynCRep_for_initialization.npz"  # Formerly called using these params
        )
        # '100_2_5.0_4_0.2_0.2_0'
        self.data_path = Path(__file__).parent / "inputs"
        self.theta = np.load(
            (self.data_path / str("theta_" + self.label)).with_suffix(".npz"),
            allow_pickle=True,
        )
        self.adj = "synthetic_data_for_DynCRep.dat"
        self.K = self.theta["u"].shape[1]
        with files("pgm.data.input").joinpath(self.adj).open("rb") as network:
            self.A, self.B, self.B_T, self.data_T_vals = import_data(
                network.name, header=0
            )

        # Define the nodes, positions, number of nodes, and number of layers
        self.nodes = self.A[0].nodes()
        self.pos = nx.spring_layout(self.A[0])
        self.N = len(self.nodes)
        self.T = self.B.shape[0] - 1

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
            auc = flt(calculate_AUC(M_inf[l], B[l].astype("int")))
            self.assertAlmostEqual(auc, expected_aucs[l], delta=TOLERANCE_2)

    def test_running_temporal_version(self):
        # Setting to run the algorithm
        with open(PATH_FOR_INIT / ("setting_" + self.algorithm + ".yaml")) as fp:
            conf = yaml.safe_load(fp)

        # Update the configuration with the specific parameters for this test
        conf["K"] = self.K
        conf["initialization"] = 1
        conf["out_folder"] = self.folder
        conf["end_file"] = "_OUT_DynCRep"
        conf["files"] = self.data_path / ("theta_" + self.label)
        conf["constrained"] = False
        conf["undirected"] = False
        conf["eta0"] = 0.2
        conf["beta0"] = self.theta["beta"]
        self.conf = conf

        # Create an instance of the DynCRep model
        model = DynCRep(max_iter=800, num_realizations=1, plot_loglik=True)

        # Fit the model to the data
        u, v, w, eta, beta, Loglikelihood = model.fit(
            data=self.B,
            T=self.T,
            nodes=self.nodes,
            flag_data_T=0,
            ag=1.1,
            bg=0.5,
            **self.conf,
        )

        # Calculate the lambda_inf and M_inf
        lambda_inf = expected_Aija(u, v, w[0])
        M_inf = lambda_inf + eta * transpose_tensor(self.B)

        # Load expected values from YAML and assert results
        yaml_file = (
            Path(__file__).parent
            / "data"
            / "dyncrep"
            / "data_for_test_running_temporal_version.yaml"
        )
        self.assert_model_results_from_yaml(
            u, v, w, eta, beta, Loglikelihood, M_inf, self.B, yaml_file
        )
