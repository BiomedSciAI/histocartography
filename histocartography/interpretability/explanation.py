

class Explanation:
    def __init__(
            self,
            adjacency_matrix,
            node_features,
            node_importance,
            orginal_prediction,
            label,
            explanation_prediction=None
    ):
    """
    Explanation constructor: Object that is defining the explanation for a given sample.

    @TODO: this class may change in the future -- eg to include functions that post-process the explanation ?
                                                     to include visualization ?

    :param adjacency_matrix: (torch.FloatTensor) a |V| x |V| matrix describing the explanation
    :param node_features: (torch.FloatTensor) a |V| x d matrix that contains the node features of the explanation 
    :param node_importance: (torch.FloatTensor) a |V| array that contains the  (scaled-) relative importance of each node
    :param original_prediction: (torch.FloatTensor) a |C| array that contains the predicted probabilities
    :param label: (torch.LongTensor) a 1d tensor storing the label 
    """
        self.adjacency_matrix = adjacency_matrix
        self.node_features = node_features
        self.node_importance = node_importance
        self.original_prediction = original_prediction
        self.explanation_prediction = explanation_prediction 
        self.label = label
