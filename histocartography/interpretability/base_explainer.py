import torch
from copy import deepcopy
import dgl 
import networkx as nx 

from histocartography.utils.io import get_device
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph


class BaseExplainer:
    def __init__(
            self,
            model,
            config, 
            cuda=False,
            verbose=False
    ):
        """
        Base Explainer constructor 
        :param model: (nn.Module) a pre-trained model to run the forward pass 
        :param config: (dict) method-specific parameters 
        :param cuda: (bool) if cuda is enable 
        :param verbose: (bool) if verbose is enable
        """
        self.model = model
        self.config = config
        self.cuda = cuda
        self.device = get_device(self.cuda)
        self.verbose = verbose
        self.store_instance_map = load_superpx_graph(self.config['model_params']['model_type'])

    def explain(self, data, label):
        """
        Explain a graph instance
        :param data: (?) graph/image/tuple
        :param label: (int) label for the input data 
        """
        raise NotImplementedError('Implementation in sub classes.')

    def _build_pruned_graph(self, graph, keep_percentage):

        # a. extract the indices of the nodes to keep 
        node_importance = graph.ndata['node_importance']
        total_node_importance = torch.sum(node_importance)
        keep_node_importance = (total_node_importance * keep_percentage).cpu().item()
        sorted_node_importance, indices_node_importance = torch.sort(node_importance, descending=True)
        node_idx_to_keep = []
        culumative_node_importance = 0
        for node_imp, idx in zip(sorted_node_importance, indices_node_importance):
            culumative_node_importance += node_imp
            if culumative_node_importance <= keep_node_importance + 10e-3:
                node_idx_to_keep.append(idx.item())
            else:
                break
        node_idx_to_keep = sorted(node_idx_to_keep)

        # b. convert to networkx 
        networkx_graph = dgl.to_networkx(graph)

        # c. remove nodes 
        networkx_graph.remove_nodes_from([x for x in list(range(graph.number_of_nodes())) if x not in node_idx_to_keep])
        mapping = {val: idx for idx, val in enumerate(node_idx_to_keep)}
        networkx_graph = nx.relabel_nodes(networkx_graph, mapping)

        # d. convert back to DGL
        pruned_graph = dgl.DGLGraph()
        pruned_graph.add_nodes(len(node_idx_to_keep))
        pruned_from = [edge[0] for edge in networkx_graph.edges()]
        pruned_to = [edge[1] for edge in networkx_graph.edges()]
        pruned_graph.add_edges(pruned_from, pruned_to)
        pruned_graph.ndata['feat'] = graph.ndata['feat'][node_idx_to_keep, :].clone()
        pruned_graph.ndata['centroid'] = graph.ndata['centroid'][node_idx_to_keep, :].clone()
        pruned_graph.ndata['node_importance'] = graph.ndata['node_importance'][node_idx_to_keep].clone()

        return pruned_graph
