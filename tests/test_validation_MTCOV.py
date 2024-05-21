from importlib.resources import files
from pathlib import Path

import numpy as np
from tests.fixtures import BaseTest, ModelTestMixin
import yaml

from pgm.input.loader import import_data_mtcov
from pgm.model.mtcov import MTCOV

current_file_path = Path(__file__)
PATH_FOR_INIT = current_file_path.parent / "inputs/"
INIT_STR = "_for_initialization"


class MTCOVTestCase(BaseTest, ModelTestMixin):

    def setUp(self):
        """
        Set up the test case.
        """
        self.algorithm = "MTCOV"
        self.keys_in_thetaGT = ["u", "v", "w", "beta", "final_it", "maxL", "nodes"]
        self.gamma = 0.5
        self.out_folder = "outputs/"
        self.end_file = "_test"
        self.adj_name = "adj.csv"
        self.cov_name = "X.csv"
        self.ego = "source"
        self.alter = "target"
        self.egoX = "Name"
        self.attr_name = "Metadata"
        self.undirected = False
        self.flag_conv = "log"
        self.force_dense = False
        self.batch_size = None

        # Import data
        pgm_data_input_path = "pgm.data.input"
        self.A, self.B, self.X, self.nodes = import_data_mtcov(
            pgm_data_input_path,
            adj_name=self.adj_name,
            cov_name=self.cov_name,
            ego=self.ego,
            alter=self.alter,
            egoX=self.egoX,
            attr_name=self.attr_name,
            undirected=self.undirected,
            force_dense=self.force_dense,
        )

        self.Xs = np.array(self.X)

        with (
            files("pgm.data.model")
            .joinpath("setting_" + self.algorithm + ".yaml")
            .open("rb") as fp
        ):
            self.conf = yaml.safe_load(fp)

        # Saving the outputs of the tests inside the tests dir
        self.conf["out_folder"] = self.folder

        self.conf["end_file"] = (
            "_OUT_" + self.algorithm
        )  # Adding a suffix to the output files

        self.model = MTCOV()

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

    def _fit_model_to_data(self, conf):
        _ = self.model.fit(
            data=self.B,
            data_X=self.Xs,
            flag_conv=self.flag_conv,
            nodes=self.nodes,
            batch_size=self.batch_size,
            **conf,
        )
