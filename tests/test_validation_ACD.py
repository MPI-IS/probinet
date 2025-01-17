from importlib.resources import files
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from probinet.evaluation.community_detection import (
    compute_community_detection_metric,
    cosine_similarity,
)
from probinet.input.loader import build_adjacency_from_file
from probinet.model_selection.masking import (
    extract_mask_kfold,
    shuffle_indices_all_matrix,
)
from probinet.models.acd import AnomalyDetection
from probinet.utils.matrix_operations import normalize_nonzero_membership

from .constants import DECIMAL_2, DECIMAL_3, DECIMAL_4, DECIMAL_5, RANDOM_SEED_REPROD
from .fixtures import BaseTest


class ACDTestCase(BaseTest):
    def setUp(self):
        # Test case parameters
        self.algorithm = "ACD"
        self.label = (
            "GT_ACD_for_initialization.npz"  # Formerly called using these params
        )
        # '100_2_5.0_4_0.2_0.2_0'
        self.data_path = Path(__file__).parent / "inputs"
        self.theta = np.load(
            (self.data_path / str("theta_" + self.label)).with_suffix(".npz"),
            allow_pickle=True,
        )
        self.keys_in_thetaGT = list(
            self.theta.keys()
        )  # They should be ['z', 'u', 'v', 'w', 'mu', 'pi', 'nodes']
        self.adj = "synthetic_data_for_ACD.dat"
        self.K = self.theta["u"].shape[1]
        with files("probinet.data.input").joinpath(self.adj).open("rb") as network:
            self.gdata = build_adjacency_from_file(network.name, header=0)

        # Define the nodes, positions, number of nodes, and number of layers
        self.pos = nx.spring_layout(self.gdata.graph_list[0])
        self.N = len(self.gdata.nodes)
        self.T = self.gdata.adjacency_tensor.shape[0] - 1
        self.L = self.gdata.adjacency_tensor.shape[0]
        self.fold = 1

        # Create a random number generator with a specified seed
        self.rng = np.random.default_rng(seed=RANDOM_SEED_REPROD)

    def prepare_data(self):
        # Shuffle the indices of all matrices using the generated random seed
        self.indices = shuffle_indices_all_matrix(self.N, self.L, rng=self.rng)

        # Extract a mask for k-fold cross-validation using the shuffled indices
        self.mask = extract_mask_kfold(self.indices, self.N, fold=self.fold, NFold=5)

        # Count the number of non-zero elements in the 'z' array where the mask is true
        np.count_nonzero(self.theta["z"][self.mask[0]])

        # Create a copy of the 'B' array to use for training
        self.B_train = self.gdata.adjacency_tensor.copy()

        # Set the elements of the training array to 0 where the mask is true
        self.B_train[self.mask > 0] = 0

        # Redefine gdata
        self.gdata = self.gdata._replace(adjacency_tensor=self.B_train)

        # Create an input mask that is the logical NOT of the original mask
        self.mask_input = np.logical_not(self.mask)

    def assert_model_results(self, u, v, w, pi, mu, maxL, theta, data):
        # Assertions for u
        self.assertEqual(list(u.shape), data["u"]["shape"])
        self.assertAlmostEqual(np.sum(u), data["u"]["sum"], places=DECIMAL_5)

        # Assertions for v
        self.assertEqual(list(v.shape), data["v"]["shape"])
        self.assertAlmostEqual(np.sum(v), data["v"]["sum"], places=DECIMAL_5)

        # Assertions for w
        self.assertEqual(list(w.shape), data["w"]["shape"])
        self.assertAlmostEqual(np.sum(w), data["w"]["sum"], places=DECIMAL_5)

        # Assertions for pi and mu
        self.assertAlmostEqual(pi, data["pi"], places=DECIMAL_3)
        self.assertAlmostEqual(mu, data["mu"], places=DECIMAL_4)

        # Assertions for Loglikelihood
        self.assertAlmostEqual(maxL, data["maxL"], places=DECIMAL_2)

        _, cs_u = cosine_similarity(u, theta["u"])
        _, cs_v = cosine_similarity(v, theta["v"])

        # Assertions for cosine similarity
        self.assertAlmostEqual(
            cs_u, data["cosine_similarity"]["cs_u"], places=DECIMAL_2
        )
        self.assertAlmostEqual(
            cs_v, data["cosine_similarity"]["cs_v"], places=DECIMAL_2
        )

        u1 = normalize_nonzero_membership(u)
        v1 = normalize_nonzero_membership(v)

        f1_u = compute_community_detection_metric(u1, theta["u"], "f1")
        f1_v = compute_community_detection_metric(v1, theta["v"], "f1")

        # Assertions for f1 score
        self.assertAlmostEqual(f1_u, data["f1_score"]["f1_u"], places=DECIMAL_2)
        self.assertAlmostEqual(f1_v, data["f1_score"]["f1_v"], places=DECIMAL_2)

    def test_running_algorithm_from_random_init(self):
        # The next section is taken from the original code like this. This is a temporary
        # validation test. In the future, a test built from fixture will be added.

        self.prepare_data()

        model = AnomalyDetection(
            plot_loglik=True, num_realizations=1, convergence_tol=0.1
        )

        u, v, w, pi, mu, maxL = model.fit(
            self.gdata,
            K=self.K,
            undirected=False,
            initialization=0,
            assortative=True,
            constrained=False,
            ag=1.5,
            bg=10.0,
            pibr0=None,
            mupr0=None,
            flag_anomaly=True,
            fix_pibr=False,
            fix_mupr=False,
            mask=self.mask_input,
            fix_communities=False,
            out_inference=True,
            out_folder=self.folder,
            end_file=("_OUT_" + self.algorithm),
            files=(self.data_path / str("theta_" + self.label)).with_suffix(".npz"),
            rng=self.rng,
        )

        # Define the path to the data file
        data_file_path = (
            Path(__file__).parent
            / "data"
            / "acd"
            / "data_for_test_running_algorithm_from_random_init.yaml"
        )

        # Load data for comparison
        with open(data_file_path, "r") as f:
            data = yaml.safe_load(f)

        self.assert_model_results(
            u=u, v=v, w=w, pi=pi, mu=mu, maxL=maxL, theta=self.theta, data=data
        )

        # Load the data from the file
        path = Path(self.folder) / str("theta_OUT_" + self.algorithm)
        theta = np.load(path.with_suffix(".npz"))
        self.assertTrue(
            all(key in theta for key in self.keys_in_thetaGT[1:]),
            "Some keys are missing in the theta dictionary",
        )
        # TODO: fix the previous assert (it should not be done from 1 onwards, but using all the
        #  list. To fix it, I first need to talk to author to see why there is a z in theta if
        #  the models does not have it

    def test_running_algorithm_from_random_init_2(self):
        # The next section is taken from the original code like this. This is a temporary
        # validation test. In the future, a test built from fixture will be added.

        # Create a random number generator with a specified seed
        self.rng = np.random.default_rng(seed=RANDOM_SEED_REPROD + 1)

        self.prepare_data()

        model = AnomalyDetection(
            plot_loglik=True, num_realizations=1, convergence_tol=0.1
        )

        u, v, w, pi, mu, maxL = model.fit(
            self.gdata,
            K=self.K,
            undirected=False,
            initialization=0,
            assortative=True,
            constrained=False,
            ag=1.5,
            bg=10.0,
            pibr0=None,
            mupr0=None,
            flag_anomaly=True,
            fix_pibr=False,
            fix_mupr=False,
            mask=self.mask_input,
            fix_communities=False,
            out_inference=True,
            out_folder=self.folder,
            end_file=("_OUT_" + self.algorithm),
            files=(self.data_path / str("theta_" + self.label)).with_suffix(".npz"),
            rng=self.rng,
        )
        # Define the path to the data file
        data_file_path = (
            Path(__file__).parent
            / "data"
            / "acd"
            / "data_for_test_running_algorithm_from_random_init_2.yaml"
        )

        # Load data for comparison
        with open(data_file_path, "r") as f:
            data = yaml.safe_load(f)

        # Check results
        self.assert_model_results(
            u=u, v=v, w=w, pi=pi, mu=mu, maxL=maxL, theta=self.theta, data=data
        )

        # TODO: Add a check for the parameters stored in theta

    # @unittest.skip("Randomization seems to have a problem, random seeds might not be fixed.")
    def test_running_algorithm_from_file(self):
        # The next section is taken from the original code like this. This is a temporary
        # validation test. In the future, a test built from fixture will be added.
        self.prepare_data()
        self.pibr0 = 1e-5
        self.mupr0 = 1e-5

        model = AnomalyDetection(
            plot_loglik=True, num_realizations=1, convergence_tol=0.1
        )

        u, v, w, pi, mu, maxL = model.fit(
            self.gdata,
            K=self.K,
            undirected=False,
            initialization=1,
            assortative=True,
            constrained=False,
            ag=1.5,
            bg=10.0,
            pibr0=self.pibr0,
            mupr0=self.mupr0,
            flag_anomaly=True,
            fix_pibr=True,
            fix_mupr=True,
            mask=self.mask_input,
            fix_communities=False,
            out_inference=False,
            end_file=self.label,
            files=(self.data_path / str("theta_" + self.label)).with_suffix(".npz"),
            rng=self.rng,
        )

        # Define the path to the data file
        data_file_path = (
            Path(__file__).parent
            / "data"
            / "acd"
            / "data_for_test_running_algorithm_from_file.yaml"
        )

        # Load data for comparison
        with open(data_file_path, "r") as f:
            data = yaml.safe_load(f)

        # Check results
        self.assert_model_results(
            u=u, v=v, w=w, pi=pi, mu=mu, maxL=maxL, theta=self.theta, data=data
        )

    def test_force_dense_false(self):
        """Test the import data function with force_dense set to False, i.e., the data is sparse."""

        with files("probinet.data.input").joinpath(self.adj).open("rb") as network:
            self.gdata = build_adjacency_from_file(
                network.name, header=0, force_dense=False
            )

        model = AnomalyDetection(
            plot_loglik=True, num_realizations=1, convergence_tol=0.1
        )

        model.fit(
            self.gdata,
            K=self.K,
            out_inference=False,
            rng=self.rng,
        )

        # Assert that the maxL is right
        self.assertAlmostEqual(model.maxL, -33272.4811, places=3)
