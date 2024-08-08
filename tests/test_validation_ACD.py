from importlib.resources import files
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from pgm.input.loader import import_data
from pgm.input.tools import normalize_nonzero_membership
from pgm.model.acd import AnomalyDetection
from pgm.model_selection.masking import extract_mask_kfold, shuffle_indices_all_matrix
from pgm.model_selection.metrics import evalu
from pgm.output.evaluate import cosine_similarity

from .constants import DECIMAL_2, DECIMAL_3, DECIMAL_4, DECIMAL_5
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
        )  # They should be ['z', 'u', 'v', 'w', 'mu',
        # 'pi', 'nodes']
        self.adj = "synthetic_data_for_ACD.dat"
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
        self.L = self.B.shape[0]
        self.fold = 1
        self.seed = 0

    def prepare_data(self):
        # Create a random number generator with a specified seed
        self.prng = np.random.RandomState(seed=self.seed)

        # Generate a random integer from 0 to 1000 using the random number generator
        self.rseed = self.prng.randint(1000)

        # Shuffle the indices of all matrices using the generated random seed
        self.indices = shuffle_indices_all_matrix(self.N, self.L, rseed=self.rseed)

        # Extract a mask for k-fold cross-validation using the shuffled indices
        self.mask = extract_mask_kfold(self.indices, self.N, fold=self.fold, NFold=5)

        # Count the number of non-zero elements in the 'z' array where the mask is true
        np.count_nonzero(self.theta["z"][self.mask[0]])

        # Create a copy of the 'B' array to use for training
        self.B_train = self.B.copy()

        # Set the elements of the training array to 0 where the mask is true
        self.B_train[self.mask > 0] = 0

        # Create an input mask that is the logical NOT of the original mask
        self.mask_input = np.logical_not(self.mask)

        # Redefine the random seed to this fixed value (10)
        self.rseed = 10

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

        f1_u = evalu(u1, theta["u"], "f1")
        f1_v = evalu(v1, theta["v"], "f1")

        # Assertions for f1 score
        self.assertAlmostEqual(f1_u, data["f1_score"]["f1_u"], places=DECIMAL_2)
        self.assertAlmostEqual(f1_v, data["f1_score"]["f1_v"], places=DECIMAL_2)

    def test_running_algorithm_from_random_init(self):

        # The next section is taken from the original code like this. This is a temporary
        # validation test. In the future, a test built from fixture will be added.

        seed = 10
        self.seed = seed
        self.prepare_data()

        model = AnomalyDetection(
            plot_loglik=True, num_realizations=1, convergence_tol=0.1
        )

        extra_params = {
            "fix_communities": False,
            "files": (self.data_path / str("theta_" + self.label)).with_suffix(".npz"),
            "out_inference": True,
            "out_folder": self.folder,
            "end_file": ("_OUT_" + self.algorithm),
            "verbose": 1,
        }

        u, v, w, pi, mu, maxL = model.fit(
            data=self.B_train,
            nodes=self.nodes,
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
            rseed=self.rseed,
            **extra_params,
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
        path = Path(self.folder) / str("theta" + extra_params["end_file"])
        theta = np.load(path.with_suffix(".npz"))
        assert all(key in theta for key in self.keys_in_thetaGT[1:]), (
            "Some keys are missing in " "the " "theta dictionary"
        )
        # TODO: fix the previous assert (it should not be done from 1 onwards, but using all the
        #  list. To fix it, I first need to talk to Hadiseh to see why there is a z in theta if
        #  the model does not have it

    def test_running_algorithm_from_random_init_2(self):

        # The next section is taken from the original code like this. This is a temporary
        # validation test. In the future, a test built from fixture will be added.

        self.prepare_data()

        model = AnomalyDetection(
            plot_loglik=True, num_realizations=1, convergence_tol=0.1
        )
        extra_params = {
            "fix_communities": False,
            "in_parameters": (self.data_path / str("theta_" + self.label)).with_suffix(
                ".npz"
            ),
            "out_inference": False,
            "end_file": self.label,
            "verbose": 1,
        }
        u, v, w, pi, mu, maxL = model.fit(
            data=self.B_train,
            nodes=self.nodes,
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
            rseed=self.rseed,
            **extra_params,
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
        extra_params = {
            "fix_communities": False,
            "files": (self.data_path / str("theta_" + self.label)).with_suffix(".npz"),
            # "fix_pibr": False,
            # "fix_mupr": False,
            "out_inference": False,
            "end_file": self.label,
            "verbose": 1,
        }
        u, v, w, pi, mu, maxL = model.fit(
            data=self.B_train,
            nodes=self.nodes,
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
            rseed=self.rseed,
            **extra_params,
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
