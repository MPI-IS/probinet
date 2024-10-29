from importlib.resources import files

from tests.constants import PATH_FOR_INIT
from tests.fixtures import BaseTest, ModelTestMixin
import yaml

from pgm.input.loader import build_adjacency_and_incidence_from_file
from pgm.model.jointcrep import JointCRep


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
    expected_likleihood = -7468.8053967272026
    places = 3

    def setUp(self):

        self.gdata = self._load_data(self.force_dense)

        with open(PATH_FOR_INIT / f"setting_{self.algorithm}.yaml") as fp:
            conf = yaml.safe_load(fp)

        conf["out_folder"] = self.folder
        conf["end_file"] = f"_OUT_{self.algorithm}"
        self.conf = conf
        self.conf["K"] = self.K
        self.L = len(self.gdata.graph_list)
        self.N = len(self.gdata.nodes)
        self.files = PATH_FOR_INIT / "theta_GT_JointCRep_for_initialization.npz"

        # Run model
        self.model = JointCRep()

    def _load_data(self, force_dense):
        with files("pgm.data.input").joinpath(self.adj).open("rb") as network:
            return build_adjacency_and_incidence_from_file(
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
            self.assertTrue(self.gdata.incidence_tensor.sum() > 0)
        else:
            self.assertTrue(self.gdata.incidence_tensor.sum() > 0)

    def test_force_dense_True(self):
        self.gdata = self._load_data(True)
        self.model = JointCRep()
        self._fit_model_to_data(self.conf)
        self.assertAlmostEqual(
            self.model.maxL, self.expected_likleihood, places=self.places
        )
