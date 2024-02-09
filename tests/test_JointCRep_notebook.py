"""
Unit tests for the JointCRep notebook.
"""
import os
from pathlib import Path
import unittest

from nbconvert.preprocessors import ExecutePreprocessor
import nbformat


class TestCRepNotebook(unittest.TestCase):
    """
    Test cases for the JointCRep notebook.
    """

    def test_notebook_execution(self):
        # Get the absolute path of the notebook dynamically
        root_path = Path(__file__).parent.parent
        notebook_path = root_path / "doc" / "source" / "tutorials" / "JointCRep.ipynb"

        # Store the current directory
        original_dir = Path.cwd()

        # Change to the directory containing the notebook
        notebook_dir = notebook_path.parent
        os.chdir(notebook_dir)

        # Load the notebook
        with open(notebook_path.name, "r", encoding="utf-8") as notebook_file:
            notebook_content = nbformat.read(notebook_file, as_version=4)

        # Initialize the ExecutePreprocessor
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

        # Execute the notebook
        ep.preprocess(notebook_content, {'metadata': {'path': '.'}})

        # Return to the original directory
        os.chdir(original_dir)
