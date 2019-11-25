import torch
import torch.nn as nn
from scipy.linalg import block_diag

from histocartography.ml.layers.gin_layer import GINLayer
from histocartography.utils.torch import masked_softmax


class DiffPoolLayer(nn.Module):

    def __init__(self,
                 config,
                 layer_id=0):

        super(DiffPoolLayer, self).__init__()
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.activation = config['activation']
        self.dropout = config['dropout']
        self.use_bn = config['use_bn']

        self.feat_gc = GINLayer(
            node_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
            act=self.activation,
            layer_id=0,
            use_bn=self.use_bn
        )

        self.pool_gc = GINLayer(
            node_dim=self.input_dim,
            hidden_dim=self.output_dim,
            out_dim=self.output_dim,
            act=self.activation,
            layer_id=0,
            use_bn=self.use_bn
        )

        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        self.reg_loss.append(EntropyLoss())

    def forward(self, g, h):
        """
        The first pooling layer is computed on batched graph.
        We first take the adjacency matrix of the batched graph, which is block-wise diagonal.
        We then compute the assignment matrix for the whole batch graph, which will also be block diagonal
        """

        feat = self.feat_gc(g, h)
        assign_tensor = self.pool_gc(g, h)
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


class EntropyLoss(nn.Module):
    # Return Scalar
    def forward(self, adj, anext, s_l):
        entropy = (torch.distributions.Categorical(
            probs=s_l).entropy()).sum(-1).mean(-1)
        assert not torch.isnan(entropy)
        return entropy
