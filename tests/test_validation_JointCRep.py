"""
This is the test module for the JointCRep algorithm.
"""

from importlib.resources import files

from tests.fixtures import BaseTest, ModelTestMixin
import yaml

from pgm.input.loader import import_data
from pgm.model.jointcrep import JointCRep

# pylint: disable=missing-function-docstring, too-many-locals, too-many-instance-attributes


class JointCRepTestCase(BaseTest, ModelTestMixin):
    """
    The basic class that inherits unittest.TestCase
    """

    def setUp(self):
        """
        Set up the test case.
        """
        # Test case parameters
        self.algorithm = "JointCRep"
        self.keys_in_thetaGT = ["u", "v", "w", "eta", "final_it", "maxL", "nodes"]
        self.adj = "synthetic_data.dat"
        self.ego = "source"
        self.alter = "target"
        self.K = 2
        self.undirected = False
        self.flag_conv = "log"
        self.force_dense = False

        # Import data: removing self-loops and making binary

        with files("pgm.data.input").joinpath(self.adj).open("rb") as network:
            self.A, self.B, self.B_T, self.data_T_vals = import_data(
                network.name,
                ego=self.ego,
                alter=self.alter,
                undirected=self.undirected,
                force_dense=self.force_dense,
                noselfloop=True,
                binary=True,
                header=0,
            )
        self.nodes = self.A[0].nodes()

        # Setting to run the algorithm

        with (
            files("pgm.data.model")
            .joinpath("setting_" + self.algorithm + ".yaml")
            .open("rb") as fp
        ):
            conf = yaml.safe_load(fp)

        # Saving the outputs of the tests inside the tests dir
        conf["out_folder"] = self.folder

        conf["end_file"] = (
            "_OUT_" + self.algorithm
        )  # Adding a suffix to the output files

        self.conf = conf

        self.conf["K"] = self.K

        self.L = len(self.A)

        self.N = len(self.nodes)

        # Run model

        self.model = JointCRep()

    # test case function to check the JointCRep.set_name function
    def test_import_data(self):

        # Check if the force_dense flag is set to True
        if self.force_dense:
            # If force_dense is True, assert that the sum of all elements in the
            # matrix B is greater than 0
            self.assertTrue(self.B.sum() > 0)
        else:
            # If force_dense is False, assert that the sum of all values in the sparse
            # matrix B is greater than 0
            self.assertTrue(self.B.vals.sum() > 0)
