import networkx as nx
import numpy as np
import dgl


def adj_to_networkx(
        adj,
        feat,
        node_importance=None,
        threshold=0.1,
        max_component=False,
        rm_iso_nodes=False,
        centroids=None,
        nuclei_labels=None):
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
    nx.set_node_attributes(graph, feat, 'feat')
    if centroids is not None:
        centroids_dict = {}
        for node_id in range(num_nodes):
            centroids_dict[node_id] = centroids[node_id, :]
        nx.set_node_attributes(graph, centroids_dict, 'centroid')

    # build topology
    adj[adj < threshold] = 0
    edge_list = np.nonzero(adj)
    weights = adj[adj > threshold]
    weighted_edge_list = [(from_.item(), to_.item(), weights[idx].item()) for (
        idx, (from_, to_)) in enumerate(edge_list)]  # if from_ <= to_
    graph.add_weighted_edges_from(weighted_edge_list)

    # set node importance
    if node_importance is not None:
        node_importance_dict = {}
        for node_id in range(num_nodes):
            node_importance_dict[node_id] = node_importance[node_id]
        nx.set_node_attributes(graph, node_importance_dict, 'node_importance')

    # set nuclei labels
    if nuclei_labels is not None:
        nuclei_labels_dict = {}
        for node_id in range(num_nodes):
            nuclei_labels_dict[node_id] = nuclei_labels[node_id]
        nx.set_node_attributes(graph, nuclei_labels_dict, 'nuclei_label')

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


def adj_to_dgl(
        adj,
        feat,
        node_importance=None,
        threshold=0.1,
        max_component=False,
        rm_iso_nodes=False,
        centroids=None,
        nuclei_labels=None):
    """Cleaning a graph by thresholding its node values.

    Args:
        - adj               :  Adjacency matrix.
        - feat              :  An array of node features.
        - threshold         :  The weight threshold.
        - max_component     :  if return the largest cc
        - rm_iso_nodes      : if remove isolated nodes
    """

    networkx_graph = adj_to_networkx(
        adj,
        feat,
        node_importance,
        threshold,
        max_component,
        rm_iso_nodes,
        centroids,
        nuclei_labels)
    graph = dgl.DGLGraph()

    node_keys = []
    for cand_key in ['node_importance', 'centroid', 'feat', 'nuclei_label']:
        try:
            nx.get_node_attributes(networkx_graph, cand_key)
            node_keys.append(cand_key)
        except BaseException:
            x = 0  # do nothing...

    graph.from_networkx(
        networkx_graph,
        edge_attrs=None,
        node_attrs=None if len(node_keys) == 0 else node_keys)
    return graph


def set_graph_on_cuda(graph):
    cuda_graph = dgl.DGLGraph()
    cuda_graph.add_nodes(graph.number_of_nodes())
    cuda_graph.add_edges(graph.edges()[0], graph.edges()[1])
    for key_graph, val_graph in graph.ndata.items():
        tmp = graph.ndata[key_graph].clone()
        cuda_graph.ndata[key_graph] = tmp.cuda()
    for key_graph, val_graph in graph.edata.items():
        cuda_graph.edata[key_graph] = graph.edata[key_graph].clone().cuda()
    return cuda_graph


def set_graph_on_cpu(graph):
    cpu_graph = dgl.DGLGraph()
    cpu_graph.add_nodes(graph.number_of_nodes())
    cpu_graph.add_edges(graph.edges()[0], graph.edges()[1])
    for key_graph, val_graph in graph.ndata.items():
        tmp = graph.ndata[key_graph].clone()
        cpu_graph.ndata[key_graph] = tmp.cpu()
    for key_graph, val_graph in graph.edata.items():
        cpu_graph.edata[key_graph] = graph.edata[key_graph].clone().cpu()
    return cpu_graph


def to_cpu(x):
    if isinstance(x, dgl.DGLGraph):
        return set_graph_on_cpu(x)


def to_device(x):
    if isinstance(x, dgl.DGLGraph):
        return set_graph_on_cuda(x)


def copy_graph(x):
    graph_copy = dgl.DGLGraph(graph_data=x)
    for k, v in x.ndata.items():
        graph_copy.ndata[k] = v.clone()
    for k, v in x.edata.items():
        graph_copy.edata[k] = v.clone()
    return graph_copy
