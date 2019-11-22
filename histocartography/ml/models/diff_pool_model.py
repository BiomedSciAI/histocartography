import torch
import torch.nn as nn
import dgl
from scipy.sparse import coo_matrix

from histocartography.ml.models.base_model import BaseModel
from histocartography.ml.layers.multi_layer_gnn import MultiLayerGNN
from histocartography.ml.layers.diff_pool_layer import DiffPoolLayer
from histocartography.utils.torch import batch2tensor
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN


class DiffPool(nn.Module):
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

        # use the config dict to set the input arguments.
        self.input_dim = input_dim
        self.hidden_dim = config['gnn_params'][0]['hidden_dim']
        self.embedding_dim = config['gnn_params'][0]['embedding_dim']
        self.num_classes = config['num_classes']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.use_bn = config['use_bn']
        self.n_pooling = len(config['pooling_params'])
        self.pool_ratio = config['pool_ratio']
        self.concat = config['cat']
        self.batch_size = batch_size
        assign_dim = int(max_num_node * self.pool_ratio) * batch_size
        self.entropy_loss = []
        assign_dims = [assign_dim]
        if self.concat:
            pool_embedding_dim = self.hidden_dim * (self.n_layers - 1) + self.embedding_dim
        else:
            pool_embedding_dim = self.embedding_dim

        # list of GNN modules before the first diffpool operation
        self.multi_level_diff_pool_layers = nn.ModuleList()
        self.multi_level_gnn_layers = nn.ModuleList()

        self.num_aggs = 1  # @TODO what is this parameter ?

        self._update_gnn_config(config['gnn_params'][0], input_dim)
        self.graph_level_gnn = MultiLayerGNN(config['gnn_params'][0])

        self._update_pooling_config(config['pooling_params'][0], assign_dim, input_dim)
        self.graph_level_pooling = DiffPoolLayer(config['pooling_params'][0])

        self._update_gnn_config(config['gnn_params'][1], input_dim)
        self.multi_level_gnn_layers.append(MultiLayerGNN(config['gnn_params'][1]))

        for i in range(self.n_pooling - 1):
            self._update_pooling_config(config['pooling_params'][i], assign_dim)
            self.multi_level_diff_pool_layers.append(
                DiffPoolLayer(config['pooling_params'][i])
            )
            self.multi_level_gnn_layers.append(
                MultiLayerGNN(config['gnn_params'][i+1])
            )
            assign_dims.append(assign_dim)

        # predicting layer
        if self.concat:
            self.pred_input_dim = pool_embedding_dim * \
                self.num_aggs * (self.n_pooling + 1)
        else:
            self.pred_input_dim = self.embedding_dim * self.num_aggs
        self.pred_layer = nn.Linear(self.pred_input_dim, self.num_classes)

    def _update_gnn_config(self, config, input_dim):
        config['input_dim'] = input_dim
        config['use_bn'] = self.use_bn
        config['dropout'] = self.dropout
        return config

    def _update_pooling_config(self, config, assign_dim, input_dim):
        # 1- compute updated parameters
        assign_dim = int(assign_dim * self.pool_ratio)

        # 2- set parameters
        config['assign_dim'] = assign_dim
        config['input_dim'] = input_dim
        config['use_bn'] = self.use_bn
        config['dropout'] = self.dropout
        return config

    def forward(self, g):
        self.entropy_loss = []
        h = g.ndata[GNN_NODE_FEAT_IN]
        # node feature for assignment matrix computation is the same as the
        # original node feature

        out_all = []

        # we use GCN blocks to get an embedding first
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
            h, adj = diffpool_layer(h, adj)
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
