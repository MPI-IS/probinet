import sys
from unittest.mock import patch

from tests.fixtures import BaseTest

from pgm.main import main


class TestLogger(BaseTest):
    @patch('pgm.model.crep.CRep.fit')
    @patch('pgm.main.configure_logger')
    def test_logger_creation(self, mock_configure_logger, mock_fit):
        sys.argv = ['main', '-a', 'CRep', '-o', str(self.folder), '-l', 'D']

        # Call the main function
        main()

        # Check that the fit is called
        mock_fit.assert_called_once()

        # Check that the logger was created
        mock_configure_logger.assert_called_once()

        # Get the namespace from the call
        namespace_arg = mock_configure_logger.call_args[0][0]

        # Check that D is the log level
        assert namespace_arg.log_level == 'D'

        # Check that the log file is None
        assert namespace_arg.log_file is None
