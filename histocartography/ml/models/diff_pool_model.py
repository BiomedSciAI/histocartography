import torch
import torch.nn as nn
import dgl

from histocartography.ml.layers.multi_layer_gnn import MultiLayerGNN
from histocartography.ml.layers.diff_pool_layer import DiffPoolLayer
from histocartography.ml.layers.dense_diff_pool_layer import DenseDiffPoolLayer
from histocartography.utils.torch import batch2tensor
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN
from histocartography.ml.models.base_model import BaseModel


class DiffPool(BaseModel):
    """
    Implementation of the Differentiable Pooling (DiffPool) algortihms proposed by Ying et al.
    (https://arxiv.org/abs/1806.08804).

    DiffPool is implementing a learned pooling mechanism that is iteratively coarsening the original graph. The
    information extracted at the different scales (ie the different graphs) is then used for classifying the graph.

    Implementation: DiffPool is using 2 types of modules, a GNN layer and a pooling layer. Therefore if we seek
    to build 2 pooling levels we have the following:

    - 1 graph level GNN: regular GNN operating on the original graph to build node embeddings (defined by g, h)
    - 1 graph level pooling: pool the original graph to create a 1st pooled graph (defined by adj, h)
    - 1 1-pooled graph GNN: GNN operating on the 1-pooled graph
    - 1 1-pooled pooling: pool the 1-pooled graph to create the 2-pooled graph
    - 1 2-pooled graph GNN: GNN operating on the 2-pooled graph

    => we end up with node features on 3 graphs: original, 1-pooled, 2-pooled. We can then have a readout function
    that is aggregating this information to classify the graph itself.

    """

    def __init__(self, config, input_dim, max_num_node, batch_size):
        """

        :param config: (dict)
        :param input_dim: data specific parameter. Known at run time.
        :param max_num_node: data specific parameter. Known at run time
        :param batch_size: learning param .
        """

        super(DiffPool, self).__init__()

        # 1- set class attributes
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.config = config
        self.max_num_nodes = max_num_node

        self.num_classes = config['num_classes']
        self.num_aggs = config['num_aggs']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.dropout = config['dropout']
        self.use_bn = config['use_bn']
        self.pool_ratio = config['pool_ratio']
        self.concat = config['cat']
        self.n_pooling = len(config['pooling_params'])

        # 2- build diff pool layers
        self._build_diff_pool()

        # 3- build classification
        self._build_classification()

    def _build_classification(self):
        # predicting layer
        if self.concat:
            self.pred_input_dim = self.config['pooling_params'][-1]['input_dim'] * \
                self.num_aggs * self.n_pooling + \
                self.config['gnn_params'][-1]['input_dim'] * (self.config['gnn_params'][-1]['n_layers'] + 1) * self.num_aggs
        else:
            self.pred_input_dim = self.config['pooling_params'][-1]['input_dim'] * self.num_aggs
        self.pred_layer = nn.Linear(self.pred_input_dim, self.num_classes)

    def _build_diff_pool(self):
        assign_dim = int(self.max_num_nodes * self.pool_ratio) * self.batch_size
        self.entropy_loss = []
        self.assign_dims = [assign_dim]

        self._update_gnn_config(self.config['gnn_params'][0],
                                input_dim=self.input_dim)
        self.graph_level_gnn = MultiLayerGNN(self.config['gnn_params'][0])

        self._update_pooling_config(self.config['pooling_params'][0],
                                    assign_dim,
                                    input_dim=self.config['gnn_params'][0]['output_dim'],
                                    n_prev_layers=self.config['gnn_params'][0]['n_layers'],
                                    prev_input_dim=self.input_dim)

        self.graph_level_pooling = DiffPoolLayer(self.config['pooling_params'][0])

        self.multi_level_diff_pool_layers = nn.ModuleList()
        self.multi_level_gnn_layers = nn.ModuleList()

        self._update_gnn_config(self.config['gnn_params'][1],
                                input_dim=self.config['gnn_params'][0]['output_dim'])
        self.multi_level_gnn_layers.append(MultiLayerGNN(self.config['gnn_params'][1]))

        for i in range(1, self.n_pooling):
            self._update_pooling_config(self.config['pooling_params'][i],
                                        assign_dim,
                                        input_dim=self.config['gnn_params'][i]['output_dim'],
                                        n_prev_layers=self.config['gnn_params'][i]['n_layers']
                                        )
            self.multi_level_diff_pool_layers.append(
                DenseDiffPoolLayer(self.config['pooling_params'][i])
            )

            self._update_gnn_config(self.config['gnn_params'][i + 1],
                                    input_dim=self.config['gnn_params'][i]['output_dim'])
            self.multi_level_gnn_layers.append(
                MultiLayerGNN(self.config['gnn_params'][i + 1])
            )

            self.assign_dims.append(self.config['pooling_params'][i]['output_dim'])

    def _update_gnn_config(self, config, input_dim):
        config['input_dim'] = input_dim
        config['hidden_dim'] = self.hidden_dim
        config['output_dim'] = self.output_dim
        config['use_bn'] = self.use_bn
        config['dropout'] = self.dropout
        return config

    def _update_pooling_config(self, config, assign_dim, input_dim, n_prev_layers, prev_input_dim):
        # 1- compute updated parameters
        assign_dim = int(assign_dim * self.pool_ratio)

        # 2- set parameters
        config['output_dim'] = assign_dim
        config['hidden_dim'] = self.hidden_dim
        if self.concat:
            config['input_dim'] = input_dim * n_prev_layers + prev_input_dim
        else:
            config['input_dim'] = input_dim
        config['use_bn'] = self.use_bn
        config['dropout'] = self.dropout
        return config

    def forward(self, g):
        self.entropy_loss = []
        out_all = []

        h = g.ndata[GNN_NODE_FEAT_IN]

        g_embedding = self.graph_level_gnn(g, h, self.concat)
        g.ndata[GNN_NODE_FEAT_IN] = g_embedding

        readout = dgl.sum_nodes(g, GNN_NODE_FEAT_IN)
        out_all.append(readout)
        if self.num_aggs == 2:
            readout = dgl.max_nodes(g, GNN_NODE_FEAT_IN)
            out_all.append(readout)

        adj, h = self.graph_level_pooling(g, g_embedding)
        node_per_pool_graph = int(adj.size()[0] / self.batch_size)

        h, adj = batch2tensor(adj, h, node_per_pool_graph)
        h = self.multi_level_gnn_layers[0](adj, h, self.concat)

        readout = torch.sum(h, dim=1)
        out_all.append(readout)
        if self.num_aggs == 2:
            readout, _ = torch.max(h, dim=1)
            out_all.append(readout)

        for i, diffpool_layer in enumerate(self.multi_level_diff_pool_layers):

            # 1. apply pooling
            h, adj = diffpool_layer(adj, h)
            # 2. apply gnn
            h = self.multi_level_gnn_layers[i + 1](adj, h, self.concat)

            readout = torch.sum(h, dim=1)
            out_all.append(readout)
            if self.num_aggs == 2:
                readout, _ = torch.max(h, dim=1)
                out_all.append(readout)

        if self.concat or self.num_aggs > 1:
            final_readout = torch.cat(out_all, dim=1)
        else:
            final_readout = readout

        ypred = self.pred_layer(final_readout)
        return ypred

    def loss(self, pred, label):
        """
        Compute cross entropy loss.
        :param pred: (FloatTensor)
        :param label: (LongTensor)
        :return: loss (FloatTensor)
        """
        #softmax + CE
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        for diffpool_layer in self.multi_level_diff_pool_layers:
            for key, value in diffpool_layer.loss_log.items():
                loss += value
        return loss
