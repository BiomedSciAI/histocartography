import networkx as nx


def adj_to_networkx(adj, feat, threshold=0.1, max_component=False, rm_iso_nodes=False):
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
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    # set node features
    nx.set_node_attributes(graph, feat, 'feats')

    # prune edges
    weighted_edge_list = [
        (i, j, adj[i, j])
        for i in range(num_nodes)
        for j in range(num_nodes)
        if adj[i, j] >= threshold
    ]

    # build topology
    graph.add_weighted_edges_from(weighted_edge_list)

    # extract largest cc
    if max_component:
        largest_cc = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_cc).copy()

    # rm isolated nodes
    if rm_iso_nodes:
        graph.remove_nodes_from(list(nx.isolates(graph)))

    return graph
