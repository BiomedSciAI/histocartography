"""Unit test for module."""
import unittest
from blueprint.module import hello_world


class ModuleTestCase(unittest.TestCase):
    """ModuleTestCase class."""

    def setUp(self):
        """Setting up the test."""
        pass

    def test_hello_world(self):
        """Test hello_world()."""
        self.assertEqual(hello_world(), 'Hello World')

    def tearDown(self):
        """Tear down the tests."""
        pass
