from torch.nn import Module

from histocartography.ml.layers.multi_layer_gnn import MultiLayerGNN


class BaseModel(Module):

    def __init__(self, config):
        """
        Base model constructor.
        """
        super(BaseModel, self).__init__()

        self.num_classes = config['num_classes']
        self.dropout = config['dropout']
        self.use_bn = config['use_bn']

    def _update_config(self, config, input_dim=None, egde_dim=None):
        """
        Update config params with data-dependent parameters
        """
        if input_dim is not None:
            config['input_dim'] = input_dim

        config['edge_dim'] = self.edge_dim
        config['use_bn'] = self.use_bn
        config['dropout'] = self.dropout

    def _build_cell_graph_params(self, config):
        """
        Build cell graph multi layer GNN
        """
        self._update_config(config, self.ll_node_dim, self.edge_dim)
        self.cell_graph_gnn = MultiLayerGNN(config=config)

    def _build_superpx_graph_params(self, superpx_config, input_dim=None, edge_dim=None):
        """
        Build super pixel multi layer GNN
        """
        if input_dim is not None:
            self._update_config(superpx_config, input_dim, edge_dim)
        self.superpx_gnn = MultiLayerGNN(config=superpx_config)

    def _build_classification_params(self):
        """
        Build classification parameters
        """
        raise NotImplementedError('Implementation in subclasses.')

    def forward(self, graphs):
        """
        Forward pass
        :param graphs:
        """
        raise NotImplementedError('Implementation in subclasses.')
