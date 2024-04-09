"""
This file contains the fixtures for the tests.
"""
import tempfile
import unittest

rtol = 1e-2
decimal = 5


class BaseTest(unittest.TestCase):
    def run(self, result=None):
        # Create a temporary directory for the duration of the test
        with tempfile.TemporaryDirectory() as temp_output_folder:
            # Store the path to the temporary directory in an instance variable
            self.folder = temp_output_folder + '/'
            # Call the parent class's run method to execute the test
            super().run(result)