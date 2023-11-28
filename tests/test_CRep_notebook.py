import os
import unittest

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class TestCRepNotebook(unittest.TestCase):

    def test_notebook_execution(self):
        # Get the absolute path of the notebook dynamically
        notebook_path = "doc/source/tutorials/CRep.ipynb"

        # Change to the directory containing the notebook
        notebook_dir = os.path.dirname(notebook_path)
        os.chdir(notebook_dir)

        # Load the notebook
        with open("CRep.ipynb", "r", encoding="utf-8") as notebook_file:
            notebook_content = nbformat.read(notebook_file, as_version=4)

        # Initialize the ExecutePreprocessor
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

        # Execute the notebook
        try:
            ep.preprocess(notebook_content, {'metadata': {'path': '.'}})
        except Exception as e:
            # If any exception occurs during execution, fail the test
            self.fail(f"Notebook execution failed: {e}")
