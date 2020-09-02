from histocartography.utils.visualization import GraphVisualization, overlay_mask
from histocartography.utils.io import save_image, write_json
from histocartography.interpretability.constants import EXPLANATION_TYPE_SAVE_SUBDIR
from histocartography.dataloader.constants import get_label_to_tumor_type, get_number_of_classes
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph

import os
import numpy as np

BASE_OUTPUT_PATH = '/dataT/pus/histocartography/Data/explainability/output'


class BaseExplanation:
    def __init__(
            self,
            config,
            image,
            image_name,
            label
    ):
        """
        Explanation constructor: Object that is defining the explanation for a given sample.

        :param image: (PIL.Image) self explicit
        :param original_prediction: (torch.FloatTensor) a |C| array that contains the predicted probabilities
        :param label: (torch.LongTensor) a 1d tensor storing the label 
        """
        self.config = config
        self.image = image[0]
        self.image_name = image_name[0]
        self.label = label

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):
        raise NotImplementedError('Implementation in subclasses')

    def draw(self):
        raise NotImplementedError('Implementation in subclasses')

    def evaluate(self):
        raise NotImplementedError('Implementation in subclasses')


class GraphExplanation(BaseExplanation):
    def __init__(
            self,
            config,
            image, 
            image_name,
            label,
            explanation_graphs,
    ):
        """
        Graph Explanation constructor: Object that is defining a graph explanation for a given sample.

        :param image: (PIL.Image) self explicit
        :param adjacency_matrix: (torch.FloatTensor) a |V| x |V| matrix describing the explanation
        :param node_features: (torch.FloatTensor) a |V| x d matrix that contains the node features of the explanation 
        :param node_importance: (torch.FloatTensor) a |V| array that contains the  (scaled-) relative importance of each node
        :param original_prediction: (torch.FloatTensor) a |C| array that contains the predicted probabilities
        :param label: (torch.LongTensor) a 1d tensor storing the label 

        @TODO: include nuclei annotations (should be returned by the PASCALE dataloder)

        """

        super(GraphExplanation, self).__init__(config, image, image_name, label)

        self.explanation_graphs = explanation_graphs
        self.num_classes = get_number_of_classes(config['model_params']['class_split'])
        self.graph_type = config['model_params']['model_type'].replace('_model', '')

        self.save_path = os.path.join(
            BASE_OUTPUT_PATH,
            'gnn',
            str(self.num_classes) + '_class_scenario',
            self.graph_type,
            EXPLANATION_TYPE_SAVE_SUBDIR[config['explanation_type']]
        )

        os.makedirs(self.save_path, exist_ok=True)

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):

        # 1. write image 
        explanation_as_image = self.draw()
        save_image(os.path.join(self.save_path, self.image_name + '_explanation.png'), explanation_as_image)

        # 2. write json 
        self._encapsulate_explanation()
        write_json(os.path.join(self.save_path, self.image_name + '_explanation.json'), self.explanation_as_dict)

    def _encapsulate_explanation(self):
        self.explanation_as_dict = {}

        # a. store config file
        self.explanation_as_dict['config'] = self.config

        # b. output 
        self.explanation_as_dict['output'] = {}
        self.explanation_as_dict['output']['label_index'] = self.label.item()
        self.explanation_as_dict['output']['label'] = get_label_to_tumor_type(self.config['model_params']['class_split'])[self.label.item()]

        # 3-d explanation graph properties
        self._remove_keys(['instance_map'])  # append all the (heavy) keys to remove from the explanation that don't need to be saved on the json 
        self.explanation_as_dict['output']['explanation'] = self.explanation_graphs

    def _remove_keys(self, keys):
        for key in keys:
            for k1, v1 in self.explanation_graphs.items():
                if key in list(v1.keys()):
                    del v1[key]

    def draw(self):
        """
        Draw explanation on the image 
        """
        visualizer = GraphVisualization(save=False)
        explanation_as_image = visualizer(
            show_cg=load_cell_graph(self.config['model_params']['model_type']),
            show_sg=load_superpx_graph(self.config['model_params']['model_type']),
            data=[self.explanation_graphs[1], self.image, self.image_name],
            node_importance=self.explanation_graphs[1]['node_importance']
        )
        return explanation_as_image


class ImageExplanation(BaseExplanation):
    def __init__(
            self,
            config,
            image, 
            image_name,
            original_prediction,
            label,
            heatmap
    ):
        """
        Image Explanation constructor: Object that is defining an image explanation for a given sample.

        :param image: (PIL.Image) self explicit
        :param adjacency_matrix: (torch.FloatTensor) a |V| x |V| matrix describing the explanation
        :param node_features: (torch.FloatTensor) a |V| x d matrix that contains the node features of the explanation 
        :param node_importance: (torch.FloatTensor) a |V| array that contains the  (scaled-) relative importance of each node
        :param original_prediction: (torch.FloatTensor) a |C| array that contains the predicted probabilities
        :param label: (torch.LongTensor) a 1d tensor storing the label 
        :param heatmap: (?) whatever is returned by torchcam 
        """

        super(ImageExplanation, self).__init__(config, image, image_name, original_prediction, label)

        self.heatmap = heatmap
        self.save_path = '/dataT/pus/histocartography/Data/explainability/output/cnn'

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):
        explanation_as_image = self.draw()
        save_image(os.path.join(self.save_path, self.image_name + '_explanation.png'), explanation_as_image)

    def _encapsulate_explanation(self):
        self.explanation_as_dict = {}
        meta_data = {}

        # a. config file
        meta_data['config'] = self.config['explanation_params']

        # b. output 
        meta_data['output'] = {}
        meta_data['output']['label_index'] = self.label.item()
        # meta_data['output']['label_set'] = [val for key, val in label_to_tumor_type.items()]
        # meta_data['output']['label'] = label_to_tumor_type[label.item()]

    def draw(self):
        """
        Draw explanation on the image 
        """
        explanation_as_image = overlay_mask(self.image, self.heatmap)
        return explanation_as_image




