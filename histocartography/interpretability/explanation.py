from histocartography.utils.visualization import GraphVisualization, overlay_mask
from histocartography.utils.io import save_image, write_json
from histocartography.interpretability.constants import EXPLANATION_TYPE_SAVE_SUBDIR

import os
import numpy as np

BASE_OUTPUT_PATH = '/dataT/pus/histocartography/Data/explainability/output/gnn'

# list all the metrics to run in order to evaluate the explanations 
METRICS = []


class BaseExplanation:
    def __init__(
            self,
            config,
            image,
            image_name,
            original_prediction,
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
        self.original_prediction = original_prediction
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
            original_prediction,
            label,
            explanation_graph,
            explanation_prediction=None
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

        super(GraphExplanation, self).__init__(config, image, image_name, original_prediction, label)

        self.explanation_graph = explanation_graph
        self.explanation_prediction = explanation_prediction 
        self.save_path = os.path.join(
            BASE_OUTPUT_PATH,
            EXPLANATION_TYPE_SAVE_SUBDIR[config['explanation_type']]
        )

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):

        # 1. evaluate the quality of the explanation using surrogate metrics (nuclei annotations, number of nodes/edges, etc...)
        self.evaluate()

        # 2. write image 
        explanation_as_image = self.draw()
        save_image(os.path.join(self.save_path, self.image_name + '_explanation.png'), explanation_as_image)

        # 3. write json 
        self._encapsulate_explanation()
        # write_json(os.path.join(self.save_path, self.image_name + '_explanation.json'), self.explanation_as_dict)

    def _encapsulate_explanation(self):
        self.explanation_as_dict = {}

        # a. config file
        self.explanation_as_dict['config'] = self.config

        # b. output 
        self.explanation_as_dict['output'] = {}
        self.explanation_as_dict['output']['label_index'] = self.label.item()
        # meta_data['output']['label_set'] = [val for key, val in label_to_tumor_type.items()]
        # meta_data['output']['label'] = label_to_tumor_type[label.item()]

        # 3-c original graph properties
        self.explanation_as_dict['output']['original'] = {}
        self.explanation_as_dict['output']['original']['logits'] = list(np.around(self.original_prediction.cpu().detach().numpy(), 2).astype(float))
        # meta_data['output']['original']['number_of_nodes'] = explanation_graph.number_of_nodes()
        # meta_data['output']['original']['number_of_edges'] = explanation_graph.number_of_edges()
        # meta_data['output']['original']['prediction'] = label_to_tumor_type[np.argmax(orig_pred)]

        # 3-d explanation graph properties
        self.explanation_as_dict['output']['explanation'] = {}
        self.explanation_as_dict['output']['explanation']['number_of_nodes'] = self.explanation_graph.number_of_nodes()
        self.explanation_as_dict['output']['explanation']['number_of_edges'] = self.explanation_graph.number_of_edges()
        if self.explanation_prediction is not None:
            self.explanation_as_dict['output']['explanation']['logits'] = list(np.around(self.explanation_prediction.cpu().detach().numpy(), 2).astype(float))
        # meta_data['output']['explanation']['prediction'] = label_to_tumor_type[np.argmax(exp_pred)]
        # meta_data['output']['explanation']['node_importance'] = str(list(node_importance))
        # meta_data['output']['explanation']['centroids'] = str([list(centroid.cpu().numpy()) for centroid in graph_visualizer.centroid_cg])
        # meta_data['output']['explanation']['edges'] = str(list(graph_visualizer.edges_cg))

    def draw(self):
        """
        Draw explanation on the image 
        """
        visualizer = GraphVisualization(save=False)
        explanation_as_image = visualizer(
            show_cg=True,
            show_sg=False,
            show_superpx=False,
            data=[self.explanation_graph, self.image, self.image_name],
            node_importance=self.explanation_graph.ndata['node_importance'] if 'node_importance' in self.explanation_graph.ndata.keys() else None
        )
        return explanation_as_image

    def evaluate(self):
        """
        Evaluate the quality of the explanation 

        return:
            - metrics: (dict) (surrogate) metrics 
        """

        for metric in METRICS:
            metric()

        return None 



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




