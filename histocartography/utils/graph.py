import networkx as nx

MAX_NUM_EDGES = 1000


def adj_to_networkx(adj, feat, threshold=0.0001, max_component=False, rm_iso_nodes=True, centroids=None):
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
    if centroids is not None:
        centroids_dict = {}
        for node_id in range(num_nodes):
            centroids_dict[node_id] = centroids[node_id, :]
        nx.set_node_attributes(graph, centroids_dict, 'centroid')

    # prune edges
    weighted_edge_list = [
        (i, j, adj[i, j])
        for i in range(num_nodes)
        for j in range(num_nodes)
        if adj[i, j] >= threshold and i <= j 
    ]
    sorted_weighted_edge_list = sorted(weighted_edge_list, key=lambda x: x[2], reverse=True)
    if len(sorted_weighted_edge_list) > MAX_NUM_EDGES:
        sorted_weighted_edge_list = sorted_weighted_edge_list[:MAX_NUM_EDGES]

    # build topology
    graph.add_weighted_edges_from(sorted_weighted_edge_list)

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
