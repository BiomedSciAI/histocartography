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
    DiffPool model.
    @TODO: (gja) add more information about the Diff Pool implementation and orginal paper.
    """

    def __init__(self, config, max_num_node, batch_size):

        super(DiffPool, self).__init__()

        # use the config dict to set the input arguments.
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.embedding_dim = config['embedding_dim']
        self.label_dim = config['label_dim']
        self.activation = config['activation']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.use_bn = config['use_bn']
        self.n_pooling = config['n_pooling']
        self.neighbor_pooling_type = config['neighbor_pooling_type']
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

        # @TODO: renaming + reorganisation of the Diff Pool parameters:

        # 1-  gc_before_pool --> graph_level_gnn : transform raw node features in node embeddings.
        # 2-  first_diffpool_layer --> diffpool_layers : 1st level diff pool layer that returns a new graph defined
        #                                               by an adjacency matrix and node features.
        # 3-  diffpool_layers --> diffpool_layers: ie merge with the 1st level diff pool layer.
        # 4-  gc_after_pool --> : new GNN operating on the pooled graphs (based on Dense GIN Layer)
        #
        # 5- Organise the config file as 2 lists. One for the GNN parameters and one for the Pooling parameters
        # 6- How to handle the parameters that known only at run time ?
        #     => have some _update_gnn_config param and _update_pooling_params

        # @TODO: Research question:
        # 1-How to include edge features in the pooling mechanism ?

        # list of GNN modules before the first diffpool operation
        self.multi_level_diff_pool_layers = nn.ModuleList()
        self.multi_level_gnn_layers = nn.ModuleList()

        # list of list of GNN modules, each list after one diffpool operation
        self.bn = True
        self.num_aggs = 1

        self.gc_before_pool = MultiLayerGNN(config['gnn_before'])

        self.first_diffpool_layer = DiffPoolLayer(
            input_dim=pool_embedding_dim,
            assign_dim=assign_dim,
            output_feat_dim=self.hidden_dim,
            activation=self.activation,
            dropout=self.dropout)

        self.multi_level_gnn_layers.append(MultiLayerGNN(config['gnn_after']))

        assign_dim = int(assign_dim * self.pool_ratio)
        for i in range(self.n_pooling - 1):
            self.multi_level_diff_pool_layers.append(
                DiffPoolLayer(
                    pool_embedding_dim,
                    assign_dim,
                    self.hidden_dim))
            gc_after_per_pool = MultiLayerGNN(config)
            self.multi_level_gnn_layers.append(gc_after_per_pool)
            assign_dims.append(assign_dim)
            assign_dim = int(assign_dim * self.pool_ratio)

        # predicting layer
        if self.concat:
            self.pred_input_dim = pool_embedding_dim * \
                self.num_aggs * (self.n_pooling + 1)
        else:
            self.pred_input_dim = self.embedding_dim * self.num_aggs
        self.pred_layer = nn.Linear(self.pred_input_dim, self.label_dim)

    # @TODO: set all the parameters that we could be know only at run time.
    def _update_gnn_config(self, config, param1, param2):
        return config

    def _update_pooling_config(self, config, param1, param2):
        return config

    def forward(self, g):
        self.entropy_loss = []
        h = g.ndata[GNN_NODE_FEAT_IN]
        # node feature for assignment matrix computation is the same as the
        # original node feature

        out_all = []

        # we use GCN blocks to get an embedding first
        g_embedding = self.gc_before_pool(g, h, self.concat)
        g.ndata[GNN_NODE_FEAT_IN] = g_embedding

        readout = dgl.sum_nodes(g, GNN_NODE_FEAT_IN)
        out_all.append(readout)
        if self.num_aggs == 2:
            readout = dgl.max_nodes(g, GNN_NODE_FEAT_IN)
            out_all.append(readout)

        adj, h = self.first_diffpool_layer(g, g_embedding)
        node_per_pool_graph = int(adj.size()[0] / self.batch_size)

        h, adj = batch2tensor(adj, h, node_per_pool_graph)
        h = self.multi_level_gnn_layers[0](adj, h, self.concat)

        readout = torch.sum(h, dim=1)
        out_all.append(readout)
        if self.num_aggs == 2:
            readout, _ = torch.max(h, dim=1)
            out_all.append(readout)

        for i, diffpool_layer in enumerate(self.multi_level_diff_pool_layers):
            h, adj = diffpool_layer(h, adj)

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
        '''
        loss function
        '''
        #softmax + CE
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        for diffpool_layer in self.multi_level_diff_pool_layers:
            for key, value in diffpool_layer.loss_log.items():
                loss += value
        return loss
