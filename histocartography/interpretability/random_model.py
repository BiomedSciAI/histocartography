import torch
import numpy as np

from histocartography.utils.io import get_device
from ..ml.layers.constants import GNN_NODE_FEAT_IN


class RandomModel:
    def __init__(
        self,
        model,
        cuda=False,
    ):
        """
        Randomly drop nuclei and test the model on the sampled input.
        :param model:
        :param adj:
        :param x:
        :param keep_prob:
        :param cuda:
        """

        super(RandomModel, self).__init__()

        # set model
        self.model = model

        # set model parameters
        self.cuda = cuda
        self.device = get_device(self.cuda)

    def _masked_node_feats(self, x, num_nodes, keep_prob):
        node_mask = torch.FloatTensor(
            np.random.choice(
                a=[0, 1],
                size=(num_nodes,),
                p=[1-keep_prob, keep_prob]
            )
        )
        x = x * torch.stack(x.shape[-1]*[node_mask], dim=1).unsqueeze(dim=0)
        return x 

    def run(self, graph, keep_prob=0.1):

        print('Running with keep probability:', keep_prob)

        # extract dense adjacency and node features
        adj = graph.adjacency_matrix().to_dense().unsqueeze(dim=0)
        adj = torch.tensor(adj, dtype=torch.float).to(self.device)

        x = graph.ndata[GNN_NODE_FEAT_IN].unsqueeze(dim=0)
        x = torch.tensor(x, dtype=torch.float).to(self.device)

        # get init probs
        init_logits, _ = self.model([adj, x])
        init_probs = torch.nn.Softmax()(init_logits).cpu().detach().numpy()

        # get masked probs
        masked_adj = adj
        masked_x = self._masked_node_feats(
            x=x,
            num_nodes=adj.size()[1],
            keep_prob=keep_prob
        )
        graph = [masked_adj, masked_x]
        logits, embeddings = self.model(graph)
        probs = torch.nn.Softmax()(logits).cpu().detach().numpy()

        return masked_adj.squeeze(), masked_x.squeeze(), init_probs, probs
