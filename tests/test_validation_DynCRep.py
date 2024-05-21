from importlib.resources import files
from pathlib import Path

import networkx as nx
import numpy as np
from tests.fixtures import BaseTest, expected_Aija, flt, TOLERANCE_2
import yaml

from pgm.input.loader import import_data
from pgm.input.tools import transpose_tensor
from pgm.model.dyncrep import DynCRep
from pgm.model.dyncrep_static import CRepDyn
from pgm.output.evaluate import calculate_AUC


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

    def test_running_temporal_version(self):

        # Setting to run the algorithm
        with (
            files("pgm.data.model")
            .joinpath("setting_" + self.algorithm + ".yaml")
            .open("rb") as fp
        ):
            conf = yaml.safe_load(fp)

        # Update the configuration with the specific parameters for this test
        conf["K"] = self.K
        conf["initialization"] = 1
        # Saving the outputs of the tests into the temp folder created in the BaseTest
        conf["out_folder"] = self.folder
        conf["end_file"] = "_OUT_DynCRep"  # Adding a suffix to the output files
        conf["files"] = self.data_path / ("theta_" + self.label)
        conf["constrained"] = False
        conf["undirected"] = False
        conf["eta0"] = 0.2
        conf["beta0"] = self.theta["beta"]
        self.conf = conf

        # Create an instance of the DynCRep model
        model = DynCRep(
            max_iter=800,
            num_realizations=1,
            plot_loglik=True,
        )

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

        # Add your assertions here
        self.assertEqual(u.shape, (100, 2))
        self.assertAlmostEqual(np.sum(u), 40.00025694829692, places=3)

        # Assertions for v
        self.assertEqual(v.shape, (100, 2))
        self.assertAlmostEqual(np.sum(v), 40.001933007145794, places=3)

        # Assertions for w
        self.assertEqual(w.shape, (5, 2, 2))
        self.assertAlmostEqual(np.sum(w), 3.0039155951245258, places=3)

        # Assertions for eta and beta
        self.assertAlmostEqual(eta, 0.21687084165382248, places=3)
        self.assertAlmostEqual(beta, 0.20967743180393628, places=3)

        # Assertions for Loglikelihood
        self.assertAlmostEqual(Loglikelihood, -2872.6923935067616, places=3)

        # Assertions for AUC
        expected_aucs = [0.811, 0.829, 0.841, 0.842, 0.843]

        # Calculate the lambda_inf and M_inf
        lambda_inf = expected_Aija(u, v, w[0])
        M_inf = lambda_inf + eta * transpose_tensor(self.B)

        # Calculate the AUC for each layer
        for l in range(model.T + 1):
            auc = flt(calculate_AUC(M_inf[l], self.B[l].astype("int")))
            self.assertAlmostEqual(auc, expected_aucs[l], delta=TOLERANCE_2)

    def test_running_static_version(self):
        # Create an instance of the CRepDyn model
        model = CRepDyn(
            plot_loglik=True,
            verbose=1,
            N_real=5,
            beta0=0.25,
            undirected=False,
            flag_data_T=1,
            fix_beta=False,
            initialization=1,
            in_parameters=self.data_path / ("theta_" + self.label),
            max_iter=800,
            end_file=self.label,
            eta0=0.2,
            constrained=True,
            ag=1.1,
            bg=0.5,
            fix_eta=False,
        )

        # Fit the model to the data
        u, v, w, eta, beta, Loglikelihood = model.fit(
            data=self.B, T=self.T, nodes=self.nodes, K=self.K
        )

        # Add your assertions here
        self.assertEqual(u.shape, (100, 2))
        self.assertAlmostEqual(np.sum(u), 100.0, places=3)

        # Assertions for v
        self.assertEqual(v.shape, (100, 2))
        self.assertAlmostEqual(np.sum(v), 99.92123973890051, places=3)

        # Assertions for w
        self.assertEqual(w.shape, (1, 2))
        self.assertAlmostEqual(np.sum(w), 0.03792499007908572, places=3)

        # Assertions for eta and beta
        self.assertAlmostEqual(eta, 0.06141942760787744, places=3)
        self.assertAlmostEqual(beta, 0.35602236108533, places=3)

        # Assertions for Loglikelihood
        self.assertAlmostEqual(Loglikelihood, -3174.04938026765, places=3)

        # Assertions for AUC
        expected_aucs = [0.785, 0.806, 0.812, 0.816, 0.817]

        # Calculate the lambda_inf and M_inf
        lambda_inf = expected_Aija(u, v, w[0])
        M_inf = lambda_inf + eta * transpose_tensor(self.B)

        # Calculate the AUC for each layer
        for l in range(model.T + 1):
            auc = flt(calculate_AUC(M_inf[l], self.B[l].astype("int")))
            self.assertAlmostEqual(auc, expected_aucs[l], delta=TOLERANCE_2)

    def test_static_is_dynamical_with_extra_flag(self):
        # Create an instance of the CRepDyn model
        model_static = CRepDyn(
            plot_loglik=True,
            verbose=1,
            N_real=1,
            beta0=0.25,
            undirected=False,
            flag_data_T=1,
            fix_beta=False,
            initialization=1,
            in_parameters=self.data_path / ("theta_" + self.label),
            max_iter=800,
            end_file=self.label,
            eta0=0.2,
            constrained=True,
            ag=1.1,
            bg=0.5,
            fix_eta=False,
        )
        # Fit the model to the data
        u_static, v_static, w_static, eta_static, beta_static, Loglikelihood_static = (
            model_static.fit(data=self.B, T=self.T, nodes=self.nodes, K=self.K)
        )

        # Setting to run the dynamic algorithm with the extra flag
        with (
            files("pgm.data.model")
            .joinpath("setting_" + self.algorithm + ".yaml")
            .open("rb") as fp
        ):
            conf = yaml.safe_load(fp)

        # Update the configuration with the specific parameters for this test
        conf["K"] = self.K
        conf["initialization"] = 1
        # Saving the outputs of the tests into the temp folder created in the BaseTest
        conf["out_folder"] = self.folder
        conf["end_file"] = "_OUT_DynCRep"  # Adding a suffix to the output files
        conf["files"] = self.data_path / ("theta_" + self.label)
        conf["constrained"] = True
        conf["undirected"] = False
        conf["eta0"] = 0.2
        conf["beta0"] = 0.25
        conf["assortative"] = True

        self.conf = conf

        # Create an instance of the CRepDyn_w_temp model
        model_static_from_dynamic = DynCRep(
            max_iter=800,
            num_realizations=1,
            plot_loglik=True,
        )
        # Fit the model to the data
        (
            u_stat_dyn,
            v_stat_dyn,
            w_stat_dyn,
            eta_stat_dyn,
            beta_stat_dyn,
            Loglikelihood_stat_dyn,
        ) = model_static_from_dynamic.fit(
            data=self.B,
            T=self.T,
            nodes=self.nodes,
            flag_data_T=1,  # this is the important flag
            ag=1.1,
            bg=0.5,
            temporal=False,
            **self.conf,
        )

        # Assert that the output variables from the fit are the same
        self.assertTrue(
            np.allclose(u_static, u_stat_dyn),
            "The u's from static and dynamic models are different.",
        )
        self.assertTrue(
            np.allclose(v_static, v_stat_dyn),
            "The v's from static and dynamic models are different.",
        )
        self.assertTrue(
            np.allclose(w_static, w_stat_dyn),
            "The w's from static and dynamic models are different.",
        )
        self.assertTrue(
            np.allclose(eta_static, eta_stat_dyn),
            "The eta's from static and dynamic models are different.",
        )
        self.assertTrue(
            np.allclose(beta_static, beta_stat_dyn),
            "The beta's from static and dynamic models are different.",
        )
        self.assertTrue(
            np.allclose(Loglikelihood_static, Loglikelihood_stat_dyn),
            "The Loglikelihoods from static and dynamic models are different.",
        )

        # Assert that the many of the attributes of both models are the same
        self.assertTrue(
            np.isclose(model_static.maxL, model_static_from_dynamic.maxL),
            "The maxL attributes from static and dynamic models are different.",
        )
        self.assertEqual(
            model_static.max_iter,
            model_static_from_dynamic.max_iter,
            "The max_iter attributes from static and dynamic models are different.",
        )
        self.assertEqual(
            model_static.constrained,
            model_static_from_dynamic.constrained,
            "The constrained attributes from static and dynamic models are different.",
        )
        self.assertEqual(
            model_static.assortative,
            model_static_from_dynamic.assortative,
            "The assortative attributes from static and dynamic models are different.",
        )
        self.assertTrue(
            np.isclose(model_static.beta, model_static_from_dynamic.beta),
            "The beta attributes from static and dynamic models are different.",
        )
        self.assertTrue(
            np.isclose(model_static.eta, model_static_from_dynamic.eta),
            "The eta attributes from static and dynamic models are different.",
        )
        self.assertEqual(
            model_static.fix_beta,
            model_static_from_dynamic.fix_beta,
            "The fix_beta attributes from static and dynamic models are different.",
        )
        self.assertEqual(
            model_static.fix_eta,
            model_static_from_dynamic.fix_eta,
            "The fix_eta attributes from static and dynamic models are different.",
        )
        self.assertEqual(
            model_static.fix_w,
            model_static_from_dynamic.fix_w,
            "The fix_w attributes from static and dynamic models are different.",
        )
        self.assertEqual(
            model_static.flag_data_T,
            model_static_from_dynamic.flag_data_T,
            "The flag_data_T attributes from static and dynamic models are different.",
        )
        self.assertTrue(
            np.isclose(model_static.data_rho2, model_static_from_dynamic.data_rho2),
            "The data_rho2 attributes from static and dynamic models are different.",
        )
