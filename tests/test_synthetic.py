"""
Test cases for the generate_network module.
"""

from pathlib import Path
import unittest

import networkx as nx
import numpy as np
import yaml

from pgm.synthetic.syn_acd import SyntNetAnomaly
from pgm.synthetic.syn_dyncrep import SyntheticDynCRep
from pgm.synthetic.syn_rep import affinity_matrix, GM_reciprocity
from pgm.synthetic.syn_sbm import BaseSyntheticNetwork, ReciprocityMMSBM_joints

from .constants import RTOL

# pylint: disable=missing-function-docstring, too-many-locals, too-many-instance-attributes


class TestGMReciprocity(unittest.TestCase):
    """
    Test cases for the GM_reciprocity class.
    """

    def setUp(self):
        # Set up parameters for the tests
        self.N = 100
        self.K = 3

    def _run_test(self, gm, expected_values):
        # Call the respective method
        outputs = gm()
        # Unpack the outputs
        G = outputs[0]
        # Compute the reciprocity and sparsity coefficient
        Sparsity_cof = np.round(2 * G.number_of_edges() / float(G.number_of_nodes()), 3)

        reciprocity_c = np.round(nx.reciprocity(G), 3)

        # Perform assertions based on the expected values
        self.assertIsInstance(G, nx.MultiDiGraph)
        self.assertAlmostEqual(
            len(G.nodes()), expected_values["nodes"]
        )  # Number of nodes after removing nodes
        self.assertAlmostEqual(
            len(G.edges()), expected_values["edges"]
        )  # Number of edges
        self.assertAlmostEqual(
            Sparsity_cof, expected_values["sparsity_cof"]
        )  # Average degree (2E/N)
        self.assertAlmostEqual(
            reciprocity_c, expected_values["reciprocity"], places=3
        )  # Reciprocity

    def test_reciprocity_planted_network(self):
        gm = GM_reciprocity(self.N, self.K)
        gm.verbose = False  # Disable verbose output for testing

        expected_values = {
            "nodes": 72,
            "edges": 124,
            "sparsity_cof": 3.444,
            "reciprocity": 0.516,
        }
        self._run_test(gm.reciprocity_planted_network, expected_values)

    def test_reciprocity_planted_network_with_verbose(self):
        gm = GM_reciprocity(self.N, self.K)
        gm.verbose = True

        expected_values = {
            "nodes": 72,
            "edges": 124,
            "sparsity_cof": 3.444,
            "reciprocity": 0.516,
        }
        self._run_test(gm.reciprocity_planted_network, expected_values)

    def test_planted_network_cond_independent(self):
        gm = GM_reciprocity(self.N, self.K)
        gm.verbose = False  # Disable verbose output for testing

        expected_values = {
            "nodes": 98,
            "edges": 137,
            "sparsity_cof": 2.796,
            "reciprocity": 0.0292,
        }
        self._run_test(gm.planted_network_cond_independent, expected_values)

    def test_planted_network_cond_independent_with_verbose(self):
        gm = GM_reciprocity(self.N, self.K)
        gm.verbose = True

        expected_values = {
            "nodes": 98,
            "edges": 137,
            "sparsity_cof": 2.796,
            "reciprocity": 0.0292,
        }
        self._run_test(gm.planted_network_cond_independent, expected_values)

    def test_planted_network_reciprocity_only(self):
        gm = GM_reciprocity(self.N, self.K)
        gm.verbose = False  # Disable verbose output for testing

        expected_values = {
            "nodes": 26,
            "edges": 38,
            "sparsity_cof": 2.923,
            "reciprocity": 0.579,
        }
        self._run_test(gm.planted_network_reciprocity_only, expected_values)

    def test_planted_network_reciprocity_only_with_verbose(self):
        gm = GM_reciprocity(self.N, self.K)
        gm.verbose = True

        expected_values = {
            "nodes": 26,
            "edges": 38,
            "sparsity_cof": 2.923,
            "reciprocity": 0.579,
        }
        self._run_test(gm.planted_network_reciprocity_only, expected_values)

    def check_invalid_parameters(self, params, expected_message):
        with self.assertRaises(ValueError) as context:
            GM_reciprocity(**params)
        self.assertEqual(str(context.exception), expected_message)

    def test_invalid_eta(self):
        params = {"N": 100, "K": 3, "eta": -0.5}
        expected_message = "The reciprocity coefficient eta has to be in [0, 1)!"
        self.check_invalid_parameters(params, expected_message)

    def test_invalid_over(self):
        params = {"N": 100, "K": 3, "over": 1.5}
        expected_message = "The overlapping parameter has to be in [0, 1]!"
        self.check_invalid_parameters(params, expected_message)

    def test_invalid_corr(self):
        params = {"N": 100, "K": 3, "corr": 1.5}
        expected_message = "The correlation parameter corr has to be in [0, 1]!"
        self.check_invalid_parameters(params, expected_message)

    def test_invalid_Normalization(self):
        params = {"N": 100, "K": 3, "Normalization": 2}
        expected_message = (
            "The Normalization parameter can be either 0 or 1! It is used as an indicator for "
            "generating the membership matrices u and v from a Dirichlet or a Gamma "
            "distribution, respectively. It is used when there is overlapping."
        )
        self.check_invalid_parameters(params, expected_message)

    def test_invalid_structure(self):
        params = {"N": 100, "K": 3, "structure": "invalid_structure"}
        expected_message = "The structure of the affinity matrix w can be either assortative or disassortative!"
        self.check_invalid_parameters(params, expected_message)

    def test_affinity_matrix_assortative(self):
        expected_result = np.array([[0.08, 0.008], [0.008, 0.08]])
        actual_result = affinity_matrix(
            structure="assortative", N=100, K=2, a=0.1, b=0.3
        )
        np.testing.assert_allclose(actual_result, expected_result, rtol=RTOL)

    def test_affinity_matrix_disassortative(self):
        expected_result = np.array([[0.008, 0.08], [0.08, 0.008]])
        actual_result = affinity_matrix(
            structure="disassortative", N=100, K=2, a=0.1, b=0.3
        )
        np.testing.assert_allclose(actual_result, expected_result, rtol=RTOL)


class TestBaseSyntheticNetwork(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.L = 1
        self.K = 2
        self.seed = 0
        self.eta = 50
        self.out_folder = Path(__file__).parent / "data/input/synthetic/"
        self.output_net = True
        self.show_details = True
        self.show_plots = True
        self.kwargs = {}
        self.base_synthetic_network = BaseSyntheticNetwork(
            self.N,
            self.L,
            self.K,
            self.seed,
            self.eta,
            self.out_folder,
            self.output_net,
            self.show_details,
            self.show_plots,
            **self.kwargs,
        )

    def test_init(self):
        self.assertEqual(self.base_synthetic_network.N, self.N)
        self.assertEqual(self.base_synthetic_network.L, self.L)
        self.assertEqual(self.base_synthetic_network.K, self.K)
        self.assertEqual(self.base_synthetic_network.seed, self.seed)
        self.assertEqual(self.base_synthetic_network.out_folder, self.out_folder)
        self.assertEqual(self.base_synthetic_network.output_net, self.output_net)
        self.assertEqual(self.base_synthetic_network.show_details, self.show_details)
        self.assertEqual(self.base_synthetic_network.show_plots, self.show_plots)


class TestCRepDyn(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.K = 2
        self.T = 2

    def test_CRepDyn_network(self):
        # Expected values
        self.expected_number_of_edges_graph_0 = 234
        self.expected_number_of_edges_graph_1 = 233
        self.expected_number_of_edges_graph_2 = 243
        self.expected_u_sum_axis_1 = 100.0
        self.expected_u_sum_axis_0 = np.array([50.0, 50.0])
        expected_number_of_edges = [
            self.expected_number_of_edges_graph_0,
            self.expected_number_of_edges_graph_1,
            self.expected_number_of_edges_graph_2,
        ]
        # Create the CRepDyn object
        crepdyn = SyntheticDynCRep(N=self.N, K=self.K, T=self.T)
        graphs = crepdyn.generate_net()
        self.assertEqual(len(graphs), self.T + 1)
        # Check the number of nodes and edges in each graph
        for i, graph in enumerate(graphs):
            self.assertEqual(graph.number_of_nodes(), self.N)
        self.assertEqual(graph.number_of_edges(), expected_number_of_edges[i])
        # Check the sum of the membership matrix u
        self.assertTrue(np.allclose(np.sum(crepdyn.u), self.expected_u_sum_axis_1))
        # Check the sum of the membership matrix u along axis 0
        self.assertTrue(
            np.allclose(np.sum(crepdyn.u, axis=0), self.expected_u_sum_axis_0)
        )

    def test_invalid_verbose(self):
        with self.assertRaises(ValueError):
            SyntheticDynCRep(self.N, self.K, verbose=4)

    def test_invalid_eta(self):
        with self.assertRaises(ValueError):
            SyntheticDynCRep(self.N, self.K, eta=-1)

    def test_invalid_beta(self):
        with self.assertRaises(ValueError):
            SyntheticDynCRep(self.N, self.K, beta=-1)
        with self.assertRaises(ValueError):
            SyntheticDynCRep(self.N, self.K, beta=2)

    def test_invalid_structure(self):
        with self.assertRaises(ValueError):
            SyntheticDynCRep(self.N, self.K, structure="invalid_structure")

    def test_invalid_ag(self):
        with self.assertRaises(ValueError):
            SyntheticDynCRep(self.N, self.K, ag=-1, L1=False)

    def test_invalid_bg(self):
        with self.assertRaises(ValueError):
            SyntheticDynCRep(self.N, self.K, bg=-1, L1=False)

    def test_invalid_corr(self):
        with self.assertRaises(ValueError):
            SyntheticDynCRep(self.N, self.K, corr=-1)
        with self.assertRaises(ValueError):
            SyntheticDynCRep(self.N, self.K, corr=2)

    def test_invalid_over(self):
        with self.assertRaises(ValueError):
            SyntheticDynCRep(self.N, self.K, over=-1)
        with self.assertRaises(ValueError):
            SyntheticDynCRep(self.N, self.K, over=2)


class TestReciprocityMMSBM_joints(unittest.TestCase):

    def setUp(self):
        mmsbm = ReciprocityMMSBM_joints(
            eta=50,
            avg_degree=15,
            is_sparse=True,
            structure=["assortative"],
            parameters=None,
        )
        self.mmsbm = mmsbm

    @unittest.skip("Talk to Martina about why this test breaks.")
    def test_nothing(self):
        self.mmsbm.build_Y()
        pass


class TestSyntNetAnomaly(unittest.TestCase):
    def setUp(self):
        self.N = 10
        self.syn_acd = SyntNetAnomaly(N=self.N)

    def load_expected_values(self, file_path):
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    def assert_synt_net_anomaly_attributes(self, syn_acd, expected_values):
        self.assertEqual(syn_acd.N, expected_values["N"])
        self.assertEqual(syn_acd.K, expected_values["K"])
        self.assertEqual(syn_acd.m, expected_values["m"])
        self.assertEqual(syn_acd.rseed, expected_values["rseed"])
        self.assertEqual(syn_acd.label, expected_values["label"])
        self.assertEqual(syn_acd.folder, expected_values["folder"])
        self.assertEqual(
            syn_acd.output_parameters, expected_values["output_parameters"]
        )
        self.assertEqual(syn_acd.output_adj, expected_values["output_adj"])
        self.assertEqual(syn_acd.outfile_adj, expected_values["outfile_adj"])
        self.assertEqual(syn_acd.avg_degree, expected_values["avg_degree"])
        self.assertEqual(syn_acd.rho_anomaly, expected_values["rho_anomaly"])
        self.assertEqual(syn_acd.verbose, expected_values["verbose"])
        self.assertEqual(syn_acd.pi, expected_values["pi"])
        self.assertEqual(syn_acd.ExpM, expected_values["ExpM"])
        self.assertEqual(syn_acd.mu, expected_values["mu"])
        self.assertEqual(syn_acd.structure, expected_values["structure"])
        self.assertEqual(syn_acd.eta, expected_values["eta"])
        self.assertEqual(syn_acd.ag, expected_values["ag"])
        self.assertEqual(syn_acd.bg, expected_values["bg"])
        self.assertEqual(syn_acd.L1, expected_values["L1"])
        self.assertEqual(syn_acd.corr, expected_values["corr"])
        self.assertEqual(syn_acd.over, expected_values["over"])

    def assert_graph_nodes_and_membership_matrices(
        self, G, G0, syn_acd, expected_values
    ):
        self.assertEqual(list(G.nodes()), expected_values["nodes"])
        self.assertEqual(list(G0.nodes()), expected_values["nodes"])
        expected_uv = np.array(expected_values["expected_uv"])
        expected_w = np.array(expected_values["expected_w"])
        np.testing.assert_array_equal(syn_acd.u, expected_uv)
        np.testing.assert_array_equal(syn_acd.v, expected_uv)
        np.testing.assert_array_almost_equal(syn_acd.w, expected_w)

    def test_anomaly_network_PB(self):
        G, G0 = self.syn_acd.anomaly_network_PB()
        expected_values = self.load_expected_values(
            Path(__file__).parent
            / "data"
            / "synthetic"
            / "test_anomaly_network_PB.yaml"
        )
        self.assert_synt_net_anomaly_attributes(self.syn_acd, expected_values)
        self.assert_graph_nodes_and_membership_matrices(
            G, G0, self.syn_acd, expected_values
        )

    def test_anomaly_network_PB_with_parameters(self):
        syn_acd = SyntNetAnomaly(
            N=200, K=3, corr=0.9, over=0.5, structure="disassortative", L1=True
        )
        G, G0 = syn_acd.anomaly_network_PB()
        expected_values = self.load_expected_values(
            Path(__file__).parent
            / "data/synthetic/test_anomaly_network_PB_with_parameters.yaml"
        )
        self.assert_synt_net_anomaly_attributes(syn_acd, expected_values)
        self.assertEqual(G.number_of_edges(), expected_values["number_of_edges"])
        self.assertEqual(np.sort(G.nodes())[0], expected_values["nodes"][0])
        self.assertEqual(np.sort(G.nodes())[-1], expected_values["nodes"][-1])
        self.assertEqual(np.sort(G0.nodes())[0], expected_values["nodes"][0])
        self.assertEqual(np.sort(G0.nodes())[-1], expected_values["nodes"][-1])
        self.assertEqual(np.sum(syn_acd.z), expected_values["sum_z"])
        self.assertEqual(np.sum(syn_acd.u), expected_values["sum_u"])
        self.assertEqual(np.sum(syn_acd.v), expected_values["sum_v"])
        self.assertAlmostEqual(np.sum(syn_acd.w), expected_values["sum_w"])
        expected_w = np.array(expected_values["expected_w"])
        np.testing.assert_array_almost_equal(syn_acd.w, expected_w)
