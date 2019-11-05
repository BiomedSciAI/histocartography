"""Unit test for graph_building.knn_graph_builder"""
import unittest
import importlib

from histocartography.graph_building.constants import GRAPH_BUILDING_TYPE, AVAILABLE_GRAPH_BUILDERS, GRAPH_BUILDING_MODULE
from histocartography.graph_building.base_graph_builder import BaseGraphBuilder


class GNNTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""

    def test_download_file_to_local(self):
        """Test download_file_to_local()."""

        config = {
            'graph_building_type': 'knn_graph_builder'
        }

        graph_builder_type = config[GRAPH_BUILDING_TYPE]
        if graph_builder_type in list(AVAILABLE_GRAPH_BUILDERS.keys()):
            print('sdfadf', GRAPH_BUILDING_MODULE.format(graph_builder_type))
            module = importlib.import_module(
                GRAPH_BUILDING_MODULE.format(graph_builder_type)
            )
            graph_builder = getattr(module, AVAILABLE_GRAPH_BUILDERS[graph_builder_type])(config)
        else:
            raise ValueError(
                'Decoder type: {} not recognized. Options are: {}'.format(
                    graph_builder_type, list(AVAILABLE_GRAPH_BUILDERS.keys())
                )
            )

        self.assertTrue(isinstance(graph_builder, BaseGraphBuilder))

    def tearDown(self):
        """Tear down the tests."""
