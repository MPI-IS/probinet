from importlib.resources import files
from pathlib import Path

import networkx as nx
import numpy as np
from tests.fixtures import BaseTest

from pgm.input.loader import import_data
from pgm.input.tools import normalize_nonzero_membership
from pgm.model.acd import AnomalyDetection
from pgm.model.cv import cosine_similarity, evalu, extract_mask_kfold, shuffle_indices_all_matrix


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
        # Assertions for u
        self.assertEqual(u.shape, (500, 3))
        self.assertAlmostEqual(np.sum(u), 81.99922051723053, places=3)

        # Assertions for v
        self.assertEqual(v.shape, (500, 3))
        self.assertAlmostEqual(np.sum(v), 81.9992483376804, places=3)

        # Assertions for w
        self.assertEqual(w.shape, (1, 3))
        self.assertAlmostEqual(np.sum(w), 7.149161532674793, places=3)

        # Assertions for pi and mu
        self.assertAlmostEqual(pi, 0.0252523, places=8)
        self.assertAlmostEqual(mu, 1.57966724e-20, places=20)

        # Assertions for Loglikelihood
        self.assertAlmostEqual(maxL, -25659.543216198774, places=3)

        _, cs_u = cosine_similarity(u, self.theta["u"])
        _, cs_v = cosine_similarity(v, self.theta["v"])

        # Assertions for cosine similarity
        self.assertAlmostEqual(cs_u, 0.90726359534683, places=4)
        self.assertAlmostEqual(cs_v, 0.9129321051470717, places=4)

        u1 = normalize_nonzero_membership(u)
        v1 = normalize_nonzero_membership(v)

        f1_u = evalu(u1, self.theta["u"], "f1")
        f1_v = evalu(v1, self.theta["v"], "f1")

        # Assertions for f1 score
        self.assertAlmostEqual(f1_u, 0.9031, places=4)
        self.assertAlmostEqual(f1_v, 0.9111, places=4)



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
        # Assertions for u
        self.assertEqual(u.shape, (500, 3))
        self.assertAlmostEqual(np.sum(u), 82.05276217051801, places=3)

        # Assertions for v
        self.assertEqual(v.shape, (500, 3))
        self.assertAlmostEqual(np.sum(v), 82.05286433562603, places=3)

        # Assertions for w
        self.assertEqual(w.shape, (1, 3))
        self.assertAlmostEqual(np.sum(w), 7.202797500074351, places=3)

        # Assertions for pi and mu
        self.assertAlmostEqual(pi, 0.02514133, places=8)
        self.assertAlmostEqual(mu, 1.19778552e-15, places=20)

        # Assertions for Loglikelihood
        self.assertAlmostEqual(maxL, -25779.334028903784, places=3)

        _, cs_u = cosine_similarity(u, self.theta["u"])
        _, cs_v = cosine_similarity(v, self.theta["v"])

        # Assertions for cosine similarity
        self.assertAlmostEqual(cs_u, 0.9037260632199504, places=4)
        self.assertAlmostEqual(cs_v, 0.9117428253476657, places=4)

        u1 = normalize_nonzero_membership(u)
        v1 = normalize_nonzero_membership(v)

        f1_u = evalu(u1, self.theta["u"], "f1")
        f1_v = evalu(v1, self.theta["v"], "f1")

        # Assertions for f1 score
        self.assertAlmostEqual(f1_u, 0.8907, places=4)
        self.assertAlmostEqual(f1_v, 0.9116, places=4)

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

        # Assertions for u
        self.assertEqual(u.shape, (500, 3))
        self.assertAlmostEqual(np.sum(u), 82.05343619478171, places=3)

        # Assertions for v
        self.assertEqual(v.shape, (500, 3))
        self.assertAlmostEqual(np.sum(v), 82.0529474647653, places=3)

        # Assertions for w
        self.assertEqual(w.shape, (1, 3))
        self.assertAlmostEqual(np.sum(w), 7.202722450342012, places=3)

        # Assertions for pi and mu
        self.assertAlmostEqual(pi, self.pibr0, places=8)
        self.assertAlmostEqual(mu, self.mupr0, places=20)

        # Assertions for Loglikelihood
        self.assertAlmostEqual(maxL, -25779.6679, places=3)

        _, cs_u = cosine_similarity(u, self.theta["u"])
        _, cs_v = cosine_similarity(v, self.theta["v"])

        # Assertions for cosine similarity
        self.assertAlmostEqual(cs_u, 0.90407, places=4)
        self.assertAlmostEqual(cs_v, 0.91213, places=4)

        u1 = normalize_nonzero_membership(u)
        v1 = normalize_nonzero_membership(v)

        f1_u = evalu(u1, self.theta["u"], "f1")
        f1_v = evalu(v1, self.theta["v"], "f1")

        # Assertions for f1 score
        self.assertAlmostEqual(f1_u, 0.8918, places=4)
        self.assertAlmostEqual(f1_v, 0.9106, places=4)
