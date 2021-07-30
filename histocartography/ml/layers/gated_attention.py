import torch 
import torch.nn as nn 
import dgl 

from histocartography.ml.layers.constants import GNN_NODE_FEAT_OUT


class GatedAttention(nn.Module):

    def __init__(self, node_dim=64):
        super(GatedAttention, self).__init__()

        hidden_dim = int(node_dim / 4)

        self.query = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Tanh()
        )
        self.value = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.to_weights = nn.Linear(hidden_dim, 1)

    def _weighted_sum(self, nodes):
        """
        Node update function
        """
        h = nodes.data[GNN_NODE_FEAT_OUT]
        a = nodes.data['att']
        w_h = torch.mul(h, a)
        return {'weighted_feats': w_h}

    def forward(self, g):
        """Forward Gated Attention. 

        Args:
            feats (torch.FloatTensor]): Node features

        Returns:
            torch.FloatTensor: Attention weights
        """
        feats = g.ndata[GNN_NODE_FEAT_OUT]
        att = self.query(feats).mul(self.value(feats))
        g.ndata['att'] = self.to_weights(att)  
        att = dgl.softmax_nodes(g, 'att')
        g.apply_nodes(func=self._weighted_sum)
        out = dgl.sum_nodes(g, 'weighted_feats')
        return out
