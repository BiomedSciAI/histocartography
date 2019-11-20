import torch
import torch.nn as nn
import dgl

from histocartography.ml.models.base_model import BaseModel
from histocartography.ml.layers.multi_layer_gnn import MultiLayerGNN
from histocartography.ml.layers.diff_pool_layer import DiffPoolLayer
from histocartography.utils.torch import batch2tensor


class DiffPool(BaseModel):
    """
    DiffPool model.
    @TODO: (gja) add more information about the Diff Pool implementation and orginal paper.
    """

    def __init__(self, config, input_dim, hidden_dim, embedding_dim,
                 label_dim, activation, n_layers, dropout,
                 n_pooling, batch_size, aggregator_type,
                 assign_dim, pool_ratio, cat=False):

        super(DiffPool, self).__init__()

        self.concat = cat
        self.n_pooling = n_pooling
        self.batch_size = batch_size
        self.entropy_loss = []
        assign_dims = [self.assign_dim]
        if self.concat:
            pool_embedding_dim = hidden_dim * (n_layers - 1) + embedding_dim
        else:
            pool_embedding_dim = embedding_dim
        # list of GNN modules before the first diffpool operation
        self.diffpool_layers = nn.ModuleList()
        self.gc_after_pool = nn.ModuleList()

        # list of list of GNN modules, each list after one diffpool operation
        self.assign_dim = assign_dim
        self.bn = True
        self.num_aggs = 1

        self.gc_before_pool = MultiLayerGNN(config['gnn_params'])

        self.first_diffpool_layer = DiffPoolLayer(
            pool_embedding_dim,
            self.assign_dim,
            hidden_dim,
            activation,
            dropout,
            aggregator_type)

        self.gc_after_pool.append(MultiLayerGNN(config['pooling_params'][0]))

        self.assign_dim = int(self.assign_dim * pool_ratio)
        for i in range(n_pooling - 1):
            self.diffpool_layers.append(
                DiffPoolLayer(
                    pool_embedding_dim,
                    self.assign_dim,
                    hidden_dim))
            gc_after_per_pool = MultiLayerGNN(config['pooling_params'][i])
            self.gc_after_pool.append(gc_after_per_pool)
            assign_dims.append(self.assign_dim)
            self.assign_dim = int(self.assign_dim * pool_ratio)

        # predicting layer
        if self.concat:
            self.pred_input_dim = pool_embedding_dim * \
                self.num_aggs * (n_pooling + 1)
        else:
            self.pred_input_dim = embedding_dim * self.num_aggs
        self.pred_layer = nn.Linear(self.pred_input_dim, label_dim)

    def gcn_forward(self, g, h, gc_layers, cat=False):
        """
        Return gc_layer embedding cat.
        """
        block_readout = []
        for gc_layer in gc_layers[:-1]:
            h = gc_layer(g, h)
            block_readout.append(h)
        h = gc_layers[-1](g, h)
        block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=1)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def gcn_forward_tensorized(self, h, adj, gc_layers, cat=False):
        block_readout = []
        for gc_layer in gc_layers:
            h = gc_layer(h, adj)
            block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=2)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def forward(self, g):
        self.entropy_loss = []
        h = g.ndata['feat']
        # node feature for assignment matrix computation is the same as the
        # original node feature

        out_all = []

        # we use GCN blocks to get an embedding first
        g_embedding = self.gcn_forward(g, h, self.gc_before_pool, self.concat)

        g.ndata['h'] = g_embedding

        readout = dgl.sum_nodes(g, 'h')
        out_all.append(readout)
        if self.num_aggs == 2:
            readout = dgl.max_nodes(g, 'h')
            out_all.append(readout)

        adj, h = self.first_diffpool_layer(g, g_embedding)
        node_per_pool_graph = int(adj.size()[0] / self.batch_size)

        h, adj = batch2tensor(adj, h, node_per_pool_graph)
        h = self.gcn_forward_tensorized(
            h, adj, self.gc_after_pool[0], self.concat)
        readout = torch.sum(h, dim=1)
        out_all.append(readout)
        if self.num_aggs == 2:
            readout, _ = torch.max(h, dim=1)
            out_all.append(readout)

        for i, diffpool_layer in enumerate(self.diffpool_layers):
            h, adj = diffpool_layer(h, adj)
            h = self.gcn_forward_tensorized(
                h, adj, self.gc_after_pool[i + 1], self.concat)
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
        for diffpool_layer in self.diffpool_layers:
            for key, value in diffpool_layer.loss_log.items():
                loss += value
        return loss
