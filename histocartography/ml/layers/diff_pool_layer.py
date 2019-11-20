import torch
import torch.nn as nn
from scipy.linalg import block_diag

from histocartography.ml.layers.gin_layer import GINLayer
from histocartography.utils.torch import masked_softmax


class DiffPoolLayer(nn.Module):

    def __init__(self, input_dim, assign_dim, output_feat_dim,
                 activation, dropout, aggregator_type):
        super(DiffPoolLayer, self).__init__()
        self.embedding_dim = input_dim
        self.assign_dim = assign_dim
        self.hidden_dim = output_feat_dim

        # @TODO: (gja) chanhe the input arguments of the GIN Layer.
        self.feat_gc = GINLayer(
            input_dim,
            output_feat_dim,
            activation,
            dropout,
            aggregator_type
        )

        self.pool_gc = GINLayer(
            input_dim,
            assign_dim,
            activation,
            dropout,
            aggregator_type)

        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        self.reg_loss.append(self._entropy_loss())

    def _entropy_loss(self, s_l):
        entropy = (torch.distributions.Categorical(
            probs=s_l).entropy()).sum(-1).mean(-1)
        assert not torch.isnan(entropy)
        return entropy

    def forward(self, g):
        """
        The first pooling layer is computed on batched graph.
        We first take the adjacency matrix of the batched graph, which is block-wise diagonal.
        We then compute the assignment matrix for the whole batch graph, which will also be block diagonal
        """
        feat = self.feat_gc(g)
        assign_tensor = self.pool_gc(g)
        device = feat.device
        assign_tensor_masks = []
        batch_size = len(g.batch_num_nodes)
        for g_n_nodes in g.batch_num_nodes:
            mask = torch.ones((g_n_nodes,
                               int(assign_tensor.size()[1] / batch_size)))
            assign_tensor_masks.append(mask)

        mask = torch.FloatTensor(
            block_diag(
                *
                assign_tensor_masks)).to(
            device=device)
        assign_tensor = masked_softmax(assign_tensor, mask,
                                       memory_efficient=False)
        h = torch.matmul(torch.t(assign_tensor), feat)
        adj = g.adjacency_matrix(ctx=device)
        adj_new = torch.sparse.mm(adj, assign_tensor)
        adj_new = torch.mm(torch.t(assign_tensor), adj_new)

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, adj_new, assign_tensor)

        return adj_new, h
