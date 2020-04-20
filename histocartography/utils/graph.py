import networkx as nx
import numpy as np 


def adj_to_networkx(adj, feat, threshold=0.1, max_component=False, rm_iso_nodes=False, centroids=None):
    """Cleaning a graph by thresholding its node values.

    Args:
        - adj               :  Adjacency matrix.
        - feat              :  An array of node features.
        - threshold         :  The weight threshold.
        - max_component     :  if return the largest cc
        - rm_iso_nodes      : if remove isolated nodes
    """

    # build nodes
    num_nodes = adj.shape[-1]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))

    # set node features
    nx.set_node_attributes(graph, feat, 'feats')
    if centroids is not None:
        centroids_dict = {}
        for node_id in range(num_nodes):
            centroids_dict[node_id] = centroids[node_id, :]
        nx.set_node_attributes(graph, centroids_dict, 'centroid')

    # build topology 
    adj[adj < threshold] = 0
    edge_list = np.nonzero(adj)
    weights = adj[adj > threshold]
    weighted_edge_list = [(from_.item(), to_.item(), weights[idx].item()) for (idx, (from_, to_)) in enumerate(edge_list)]  # if from_ <= to_
    graph.add_weighted_edges_from(weighted_edge_list)

    # extract largest cc
    if max_component:
        largest_cc = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_cc).copy()

    # rm isolated nodes
    if rm_iso_nodes:
        iso = list(nx.isolates(graph))
        graph.remove_nodes_from(iso)
        graph = nx.convert_node_labels_to_integers(graph)

    return graph
