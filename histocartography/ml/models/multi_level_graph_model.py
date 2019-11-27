import torch
import torch.nn as nn
import dgl

from histocartography.ml.layers.multi_layer_gnn import MultiLayerGNN
from histocartography.ml.models.base_model import BaseModel


class MultiLevelGraphModel(BaseModel):
    """
    Multi-level graph model. The information for grading tumors in WSI lie at different scales. By building 2 graphs,
    one at the cell level and one at the object level (modeled with super pixels), we can extract graph embeddings
    that once combined provide a multi-scale representation of a tumor.
    """

    def __init__(self, config):
        """

        :param config: (dict)
        :param input_dim: data specific parameter. Known at run time.
        :param max_num_node: data specific parameter. Known at run time
        :param batch_size: learning param .
        """

        super(MultiLevelGraphModel, self).__init__()

        # 1- set class attributes
        self.config = config

        self.num_classes = config['num_classes']
        self.dropout = config['dropout']
        self.use_bn = config['use_bn']
        self.concat = config['cat']

        # 2- build cell graph params
        self._build_cell_graph_params()

        # 3- build super pixel graph params
        self._build_superpx_graph_params()

        # 4- build classification params
        self._build_classification_params()

    def _build_classification_params(self):
        # predicting layer
        if self.concat:
            self.pred_input_dim = self.config['pooling_params'][-1]['input_dim'] * \
                self.n_pooling + \
                self.config['gnn_params'][-1]['input_dim'] * (self.config['gnn_params'][-1]['n_layers'] + 1)
        else:
            self.pred_input_dim = self.config['pooling_params'][-1]['input_dim'] * self.num_aggs
        self.pred_layer = nn.Linear(self.pred_input_dim, self.num_classes)

    def _build_cell_graph_params(self):
        """
        Build cell graph multi layer GNN
        """
        self.cell_graph_gnn = MultiLayerGNN(config=self.config['gnn_params'][0])

    def _build_superpx_graph_params(self):
        """
        Build super pixel multi layer GNN
        """
        self.superpx_gnn = MultiLayerGNN(config=self.config['gnn_params'][1])

    def _build_assignment_matrix(self, cell_graph, superpx_graph):
        """
        Compute the clustering assignment matrix between the cell graph
        and the super pixel graph.
        :param cell_graph: (DGLGraph)
        :param superpx_graph: (DGLGraph)
        """

    def forward(self, cell_graph, superpx_graph, assignment_matrix=None):
        """
        Foward pass.
        :param cell_graph: (DGLGraph)
        :param superpx_graph: (DGLGraph)
        :param assignment_matrix:
        """

        final_readout = torch.FloatTensor([0.])
        ypred = self.pred_layer(final_readout)
        return ypred
