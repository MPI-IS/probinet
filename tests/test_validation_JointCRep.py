from importlib.resources import files

import numpy as np
import yaml
from tests.constants import PATH_FOR_INIT, RANDOM_SEED_REPROD
from tests.fixtures import BaseTest, ModelTestMixin

from probinet.input.loader import build_adjacency_from_file
from probinet.models.jointcrep import JointCRep


class JointCRepTestCase(BaseTest, ModelTestMixin):
    algorithm = "JointCRep"
    keys_in_thetaGT = ["u", "v", "w", "eta", "final_it", "maxL", "nodes"]
    adj = "synthetic_data.dat"
    ego = "source"
    alter = "target"
    K = 2
    undirected = False
    flag_conv = "log"
    force_dense = False
    expected_likleihood = -7471.28787
    places = 3
    max_iter = 1000

    def setUp(self):
        self.gdata = self._load_data(self.force_dense)

        with open(PATH_FOR_INIT / f"setting_{self.algorithm}.yaml") as fp:
            conf = yaml.safe_load(fp)

        conf["out_folder"] = self.folder
        conf["end_file"] = f"_OUT_{self.algorithm}"
        self.conf = conf
        self.conf["K"] = self.K
        self.conf["rng"] = np.random.default_rng(seed=RANDOM_SEED_REPROD)

        self.L = len(self.gdata.graph_list)
        self.N = len(self.gdata.nodes)
        self.files = PATH_FOR_INIT / "theta_GT_JointCRep_for_initialization.npz"

        # Run model
        self.model = JointCRep(max_iter=self.max_iter)

    def _load_data(self, force_dense):
        with files("probinet.data.input").joinpath(self.adj).open("rb") as network:
            return build_adjacency_from_file(
                network.name,
                ego=self.ego,
                alter=self.alter,
                undirected=self.undirected,
                force_dense=force_dense,
                noselfloop=True,
                binary=True,
                header=0,
            )

    def test_import_data(self):
        if self.force_dense:
            self.assertTrue(self.gdata.adjacency_tensor.sum() > 0)
        else:
            self.assertTrue(self.gdata.adjacency_tensor.sum() > 0)

    def test_force_dense_True(self):
        self.gdata = self._load_data(True)
        self.model = JointCRep()
        self._fit_model_to_data(self.conf)
        self.assertAlmostEqual(
            self.model.maxL, self.expected_likleihood, places=self.places
        )

    def test_running_algorithm_from_mixin(self):
        self.running_algorithm_from_mixin()

    def test_running_algorithm_initialized_from_file_from_mixin(self):
        self.running_algorithm_initialized_from_file_from_mixin()

    def test_model_parameter_change_with_config_file(self):
        self.model_parameter_change_with_config_file()

    def test_running_algorithm_with_new_random_seed(self):
        # Change the rng
        self.conf["rng"] = np.random.default_rng(RANDOM_SEED_REPROD + 1)

        # Change end file to use the new random seed
        self.conf["end_file"] = (
            "_OUT_" + self.algorithm + "_rseed_" + str(RANDOM_SEED_REPROD + 1)
        )

        # Fit the models to the data
        self._fit_model_to_data(self.conf)

        # Load the models results
        theta = self._load_model_results()

        # Change the suffix of the ground truth file to use the new random seed
        self.algorithm = self.algorithm + "_rseed_" + str(RANDOM_SEED_REPROD + 1)

        # Load the ground truth results
        thetaGT = self._load_ground_truth_results()

        # Asserting GT information
        self._assert_ground_truth_information(theta, thetaGT)

        # Check now that the output obtained using another random seed is different
        self.algorithm = "JointCRep"

        # Load the ground truth results from a different random seed
        thetaGT_default_random_seed = self._load_ground_truth_results()

        # Assert that the "w" values are different
        self.assertFalse(
            np.array_equal(thetaGT_default_random_seed["w"], thetaGT["w"]),
            "The 'w' values should be different for different random seeds",
        )

    def test_running_algorithm_with_rng_seed_0(self):
        # Create a random number generator with seed self.conf["rseed"] (which is equal to 0)
        self.conf["rng"] = np.random.default_rng(self.conf["rseed"])

        # Change end file to use the new random seed
        self.conf["end_file"] = (
            "_OUT_" + self.algorithm + "_rseed_" + str(self.conf["rseed"])
        )

        # Fit the models to the data
        self._fit_model_to_data(self.conf)

        # Load the models results
        theta = self._load_model_results()

        # Load the ground truth results from the default case
        self.algorithm = "JointCRep"
        thetaGT_default = self._load_ground_truth_results()

        # Assert that the results are equal
        self.assertTrue(
            np.array_equal(theta["w"], thetaGT_default["w"]),
            "The 'w' values should be equal for the same random seed",
        )
