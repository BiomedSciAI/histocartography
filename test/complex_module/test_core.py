"""Unit test for complex_module.core."""
import unittest
from blueprint.complex_module.core import salutation


class CoreTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""
        pass

    def test_salutation(self):
        """Test salutation()."""
        self.assertEqual(salutation(), 'Gruezi Mitenand')

    def tearDown(self):
        """Tear down the tests."""
        pass
