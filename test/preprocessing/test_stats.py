"""Unit test for preprocessing.tissue_mask"""
import unittest
import numpy as np
import yaml
import os 
from PIL import Image
import shutil

from histocartography import PipelineRunner


class StatsTestCase(unittest.TestCase):
    """StatsTestCase class."""

    @classmethod
    def setUpClass(self):
        self.current_path = os.path.dirname(__file__)
        self.data_path = os.path.join(self.current_path, '..', 'data')
        self.image_path = os.path.join(self.data_path, 'images')
        self.image_name = '16B0001851_Block_Region_3.jpg'
        self.graph_path = os.path.join(self.data_path, 'tissue_graphs')
        self.graph_name = '283_dcis_4.bin'
        self.out_path = os.path.join(self.data_path, 'stats_test')
        if os.path.exists(self.out_path) and os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path) 
        os.makedirs(self.out_path)

    def test_graph_diameter_with_pipeline_runner(self):
        """
        Test graph diameter with pipeline runner.
        """

        config_fname = os.path.join(self.current_path, 'config', 'stats', 'graph_diameter.yml')
        with open(config_fname, 'r') as file:
            config = yaml.load(file)
        pipeline = PipelineRunner(output_path=self.out_path, save=False, **config)
        pipeline.precompute()
        output = pipeline.run(
            name=self.graph_name.replace('.bin', ''),
            graph_path=os.path.join(self.graph_path, self.graph_name)
        )
        diameter = output['diameter']
        self.assertEqual(diameter, 6)  # check true diameter 

    def test_superpixel_counter_with_pipeline_runner(self):
        """
        Test superpixel counter with pipeline runner.
        """

        config_fname = os.path.join(self.current_path, 'config', 'stats', 'superpixel_counter.yml')
        with open(config_fname, 'r') as file:
            config = yaml.load(file)
            
        pipeline = PipelineRunner(output_path=self.out_path, save=False, **config)
        pipeline.precompute()
        output = pipeline.run(
            name=self.image_name.replace('.jpg', ''),
            image_path=os.path.join(self.image_path, self.image_name)
        )
        count = output['counter']
        self.assertEqual(count, 95)  # check true count

    def tearDown(self):
        """Tear down the tests."""


if __name__ == "__main__":
    unittest.main()
