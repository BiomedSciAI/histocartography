from histocartography.utils.visualization import GraphVisualization


class BaseExplanation:
    def __init__(
            self,
            config,
            image,
            original_prediction,
            label
    ):
    """
    Explanation constructor: Object that is defining the explanation for a given sample.

    @TODO: this class may change in the future -- eg to include functions that post-process the explanation ?
                                                     to include visualization ?

    :param image: (PIL.Image) self explicit
    :param original_prediction: (torch.FloatTensor) a |C| array that contains the predicted probabilities
    :param label: (torch.LongTensor) a 1d tensor storing the label 
    """
        self.config = config
        self.image = image
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
            original_prediction,
            label,
            adjacency_matrix,
            node_features,
            node_importance,
            explanation_prediction=None
    )
    """
    Explanation constructor: Object that is defining the explanation for a given sample.

    @TODO: this class may change in the future -- eg to include functions that post-process the explanation ?
                                                     to include visualization ?

    :param image: (PIL.Image) self explicit
    :param adjacency_matrix: (torch.FloatTensor) a |V| x |V| matrix describing the explanation
    :param node_features: (torch.FloatTensor) a |V| x d matrix that contains the node features of the explanation 
    :param node_importance: (torch.FloatTensor) a |V| array that contains the  (scaled-) relative importance of each node
    :param original_prediction: (torch.FloatTensor) a |C| array that contains the predicted probabilities
    :param label: (torch.LongTensor) a 1d tensor storing the label 
    """

    super(GraphExplanation, self).__init__(image, original_prediction, label)

    self.adjacency_matrix = adjacency_matrix
    self.node_features = node_features
    self.node_importance = node_importance 
    self.explanation_prediction = explanation_prediction 

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):
        raise NotImplementedError('Implementation in subclasses')

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

        # 3-c original graph properties
        meta_data['output']['original'] = {}
        meta_data['output']['original']['logits'] = list(np.around(orig_pred, 2).astype(float))
        meta_data['output']['original']['number_of_nodes'] = cell_graph.number_of_nodes()
        meta_data['output']['original']['number_of_edges'] = cell_graph.number_of_edges()
        meta_data['output']['original']['prediction'] = label_to_tumor_type[np.argmax(orig_pred)]

        # 3-d explanation graph properties
        meta_data['output']['explanation'] = {}
        meta_data['output']['explanation']['number_of_nodes'] = explanation.number_of_nodes()
        meta_data['output']['explanation']['number_of_edges'] = explanation.number_of_edges()
        meta_data['output']['explanation']['logits'] = list(np.around(exp_pred, 2).astype(float))
        meta_data['output']['explanation']['prediction'] = label_to_tumor_type[np.argmax(exp_pred)]
        meta_data['output']['explanation']['node_importance'] = str(list(node_importance))
        meta_data['output']['explanation']['centroids'] = str([list(centroid.cpu().numpy()) for centroid in graph_visualizer.centroid_cg])
        meta_data['output']['explanation']['edges'] = str(list(graph_visualizer.edges_cg))

    def draw(self):
        """
        Draw explanation on the image 
        """
        visualizer = GraphVisualization()


# Goal: being able to save the interpretability in a folder ?
# for each sample, we want to store:
#    - the image with the interpretability draw on it 
#    - a JSON file that encapsulates all the important information
#
