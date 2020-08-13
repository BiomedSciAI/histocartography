from histocartography.utils.visualization import GraphVisualization, overlay_mask


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
        self.image = image
        self.image_name = image_name
        self.original_prediction = original_prediction
        self.label = label

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):
        raise NotImplementedError('Implementation in subclasses')

    def draw(self):
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
        """

        super(GraphExplanation, self).__init__(config, image, image_name, original_prediction, label)

        self.explanation_graph = explanation_graph
        self.explanation_prediction = explanation_prediction 
        self.save_path = '/dataT/pus/histocartography/Data/explainability/output/gnn'

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):
        explanation_as_image = self.draw()
        save_image(os.path.join(self.save_path, self.image_name + '_explanation.png'), explanation_as_image)

    def _encapsulate_explanation(self):
        self.explanation_as_dict = {}
        meta_data = {}

        # a. config file
        meta_data['config'] = self.config

        # b. output 
        meta_data['output'] = {}
        meta_data['output']['label_index'] = self.label.item()
        # meta_data['output']['label_set'] = [val for key, val in label_to_tumor_type.items()]
        # meta_data['output']['label'] = label_to_tumor_type[label.item()]

        # 3-c original graph properties
        meta_data['output']['original'] = {}
        meta_data['output']['original']['logits'] = list(np.around(self.original_prediction, 2).astype(float))
        # meta_data['output']['original']['number_of_nodes'] = explanation_graph.number_of_nodes()
        # meta_data['output']['original']['number_of_edges'] = explanation_graph.number_of_edges()
        # meta_data['output']['original']['prediction'] = label_to_tumor_type[np.argmax(orig_pred)]

        # 3-d explanation graph properties
        meta_data['output']['explanation'] = {}
        meta_data['output']['explanation']['number_of_nodes'] = explanation_graph.number_of_nodes()
        meta_data['output']['explanation']['number_of_edges'] = explanation_graph.number_of_edges()
        # meta_data['output']['explanation']['logits'] = list(np.around(exp_pred, 2).astype(float))
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
            data=[self.explanation_graph, self.image[0], self.image_name[0]])
        return explanation_as_image

# Goal: being able to save the interpretability in a folder ?
# for each sample, we want to store:
#    - the image with the interpretability draw on it 
#    - a JSON file that encapsulates all the important information
#


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




