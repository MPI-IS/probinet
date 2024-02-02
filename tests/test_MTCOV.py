import unittest

import numpy as np

from pgm.model.mtcov import MTCOV

# pylint: disable=missing-function-docstring, missing-module-docstring, too-many-locals, too-many-instance-attributes


class TestMTCOVHelpers(unittest.TestCase):
    """
    Test cases for the MTCOV class.
    """

    def setUp(self):
        # Initialize a MTCOV object for testing
        self.mtcov = MTCOV()
        # TODO: Set up any necessary parameters or state
    # Skip this test for now

    @unittest.skip('Not implemented yet')
    def test_initialize_v_undirected(self):
        """
        Test the _initialize_v method of the MTCOV class.
        """

        # Set up
        rng = np.random.default_rng()
        nodes = [0, 1, 2]
        self.mtcov.undirected = True

        # Call the method to test
        self.mtcov._initialize_v(rng, nodes)

        # Check that v has been initialized correctly
        if self.mtcov.undirected:
            self.assertTrue(np.array_equal(self.mtcov.v, self.mtcov.u))
        else:
            self.assertTrue(np.array_equal(self.mtcov.v, self.mtcov.theta['v']))
            self.assertTrue(np.array_equal(nodes, self.mtcov.theta['nodes']))

        # Check that v has been perturbed correctly
        max_entry = np.max(self.mtcov.v)
        self.assertTrue(np.all(self.mtcov.v <= max_entry * (1 + self.mtcov.err)))

    def test_method2(self):
        # TODO: Write test for method2
        pass
