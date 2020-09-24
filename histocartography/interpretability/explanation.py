from histocartography.utils.visualization import GraphVisualization, overlay_mask
from histocartography.utils.io import save_image, write_json, buffer_plot_and_get
from histocartography.interpretability.constants import EXPLANATION_TYPE_SAVE_SUBDIR
from histocartography.dataloader.constants import get_label_to_tumor_type, get_number_of_classes
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph
from histocartography.utils.visualization import overlay_mask

import os
import numpy as np
import PIL
from PIL import Image
from captum.attr import visualization as viz
from matplotlib import cm
import io 

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

        :param config: (dict) configuration parameters
        :param image: (PIL.Image) self explicit
        :param image_name: (str) self explicit
        :param label: (torch.LongTensor) a 1d tensor storing the label 
        """
        self.config = config
        self.image = image
        self.image_name = image_name
        self.label = label
        self.num_classes = get_number_of_classes(config['model_params']['class_split'])

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

        :param config: (dict) configuration parameters
        :param image: (PIL.Image) self explicit
        :param image_name: (str) self explicit
        :param label: (torch.LongTensor) a 1d tensor storing the label 
        :param explanation_graphs: (dict) all the information relative to the explanations 
            - keep_percentage --> logits 
            - keep_percentage --> latent
            - keep_percentage --> num_nodes
            - keep_percentage --> num_edges
            - keep_percentage --> node_importance
            - keep_percentage --> centroid
        """

        super(GraphExplanation, self).__init__(config, image, image_name, label)

        self.explanation_graphs = explanation_graphs
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


IMAGE_INTER_METHOD_TO_DRAWING_FN = {
    'saliency_explainer.image_deeplift_explainer': lambda heatmap, image: overlay_mask(image, heatmap),  # lambda heatmap, image: numpy_to_pil(heatmap, image),
    'saliency_explainer.image_gradcam_explainer': lambda heatmap, image: overlay_mask(image, heatmap),
    'saliency_explainer.image_gradcampp_explainer': lambda heatmap, image: overlay_mask(image, heatmap)
}

COLORMAP = 'jet'
CMAP = cm.get_cmap(COLORMAP)

def numpy_to_pil(heatmap, image):
    fig, ax = viz.visualize_image_attr(
        heatmap,
        image,
        method="blended_heat_map",
        sign="positive",
        show_colorbar=True,
        cmap=CMAP,
        title="Overlayed DeepLift", use_pyplot=False
    )
    image = buffer_plot_and_get(fig)
    return image


class ImageExplanation(BaseExplanation):
    def __init__(
            self,
            config,
            image, 
            image_name,
            label,
            heatmap,
            logits 
    ):
        """
        Image Explanation constructor: Object that is defining an image explanation for a given sample.

        :param config: (dict) configuration parameters
        :param image: (PIL.Image) self explicit
        :param image_name: (str) self explicit
        :param label: (torch.LongTensor) a 1d tensor storing the label 
        :param heatmap: (?) whatever is returned by torchcam 
        """

        super(ImageExplanation, self).__init__(config, image, image_name, label)

        self.heatmap = heatmap
        self.logits = logits
        self.save_path = os.path.join(
            BASE_OUTPUT_PATH,
            'cnn',
            str(self.num_classes) + '_class_scenario',
            EXPLANATION_TYPE_SAVE_SUBDIR[config['explanation_type']]
        )

        os.makedirs(self.save_path, exist_ok=True)

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def draw(self):
        return IMAGE_INTER_METHOD_TO_DRAWING_FN[self.config['explanation_type']](self.heatmap, self.image)

    def write(self):

        # 1. save image 
        image = self.draw()
        save_image(os.path.join(self.save_path, self.image_name + '_explanation.png'), image)

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
        self.explanation_as_dict['output']['logits'] = self.logits

