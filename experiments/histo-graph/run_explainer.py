#!/usr/bin/env python3
"""
Script for generating explanations
"""

import importlib
import torch
import mlflow
import numpy as np
from tqdm import tqdm 

from histocartography.utils.io import read_params, write_json, complete_path
from histocartography.dataloader.pascale_dataloader import make_data_loader
from histocartography.ml.models.constants import AVAILABLE_MODEL_TYPES, MODEL_TYPE
from histocartography.interpretability.constants import AVAILABLE_EXPLAINABILITY_METHODS, INTERPRETABILITY_MODEL_TYPE_TO_LOAD_FN
from histocartography.utils.arg_parser import parse_arguments
from histocartography.ml.models.constants import load_superpx_graph, load_cell_graph
from histocartography.utils.io import get_device, flatten_dict
from histocartography.interpretability.random_model import RandomModel
from histocartography.utils.visualization import GraphVisualization, agg_and_plot_interpretation
from histocartography.dataloader.constants import get_label_to_tumor_type, CLASS_SPLIT_TO_MODEL_URL


# flush warnings
import warnings
warnings.filterwarnings("ignore")

# cuda support
CUDA = torch.cuda.is_available()
DEVICE = get_device(CUDA)



def main(args):
    """
    Train HistoGraph.
    Args:
        args (Namespace): parsed arguments.
    """

    # load config file
    config = read_params(args.config_fpath, verbose=True)

    # constants
    label_to_tumor_type = get_label_to_tumor_type(config['model_params']['class_split'])

    # extract interpretability model type
    interpretability_model_type = config['explanation_params']['explanation_type']

    # make data loaders
    dataloaders, input_feature_dims = make_data_loader(
        batch_size=1,
        num_workers=args.number_of_workers,
        path=args.data_path,
        config=config,
        class_split=config['model_params']['class_split'],
        cuda=CUDA,
        load_cell_graph=load_cell_graph(config['model_type']),
        load_superpx_graph=load_superpx_graph(config['model_type']),
        load_image=True,
        load_nuclei_seg_map=True,
        fold_id=0
    )

    # append dataset info to config
    config['data_params'] = {}
    config['data_params']['input_feature_dims'] = input_feature_dims

    # define GNN model
    if interpretability_model_type in list(AVAILABLE_EXPLAINABILITY_METHODS.keys()):
        module = importlib.import_module('histocartography.utils.io')
        model = getattr(module, INTERPRETABILITY_MODEL_TYPE_TO_LOAD_FN[interpretability_model_type])(config)
        if CUDA:
            model = model.cuda()
    else:
        raise ValueError(
            'Model: {} not recognized. Options are: {}'.format(
                model_type, list(AVAILABLE_EXPLAINABILITY_METHODS.keys())
            )
        )

    # define interpretability model 
    config['explanation_params']['model_params']['class_split'] = config['model_params']['class_split']
    if interpretability_model_type in list(AVAILABLE_EXPLAINABILITY_METHODS.keys()):
        module = importlib.import_module(
            'histocartography.interpretability.{}'.format(interpretability_model_type)
        )
        interpretability_model = getattr(
            module, AVAILABLE_EXPLAINABILITY_METHODS[interpretability_model_type])(
                model, config['explanation_params']
            )
    else:
        raise ValueError(
            'Interpretability method: {} not recognized. Options are: {}'.format(
                interpretability_model_type, list(AVAILABLE_EXPLAINABILITY_METHODS.keys())
            )
        )

    # mlflow log parameters
    inter_config = flatten_dict(config['explanation_params'])
    for key, val in inter_config.items():
        mlflow.log_param(key, val)

    # explain instance from the train set
    for data, label in tqdm(dataloaders[args.split]):

        # cell_graph = data[0]

        # # Write config and base properties
        # meta_data = {}

        # # 3-a config file
        # meta_data['config'] = config['explanation_params']  # @TODO set all the params from the config file...
        # meta_data['config']['number_of_epochs'] = args.epochs
        # meta_data['config']['learning_rate'] = args.learning_rate
        # meta_data['output'] = {}
        # # 3-b label
        # meta_data['output']['label_index'] = label.item()
        # meta_data['output']['label_set'] = [val for key, val in label_to_tumor_type.items()]
        # meta_data['output']['label'] = label_to_tumor_type[label.item()]

        # try:

        # 1. Run explainer
        # adj, feats, orig_pred, exp_pred, node_importance = interpretability_model.explain(
        #     data=data,
        #     label=label
        # )

        explanation = interpretability_model.explain(
            data=data,
            label=label
        )

        explanation.write()

        # # a. visualize the original graph
        # show_cg_flag = load_cell_graph(config['model_type'])
        # show_sp_flag = load_superpx_graph(config['model_type'])
        # show_sp_map = False
        # graph_visualizer = GraphVisualization(save_path=args.out_path, show_centroid=True, show_edges=True)
        # graph_visualizer(show_cg_flag, show_sp_flag, show_sp_map, data, 1)

        # # b. visualize the explanation graph
        # graph_visualizer = GraphVisualization(save_path=args.out_path, show_centroid=False, show_edges=True)
        # instance_map = data[-1][0]
        # pruned_instance_map = np.zeros(instance_map.shape)
        # for node_id in np.nonzero(node_idx):
        #     pruned_instance_map += instance_map * (instance_map == node_id.item() + 1)
        # tmp_data = (explanation, data[1], [data[2][0] + '_explanation'], [pruned_instance_map])
        # graph_visualizer(show_cg_flag, show_sp_flag, show_sp_map, tmp_data, 1, node_importance=None)

        # # 3-c original graph properties
        # meta_data['output']['original'] = {}
        # meta_data['output']['original']['logits'] = list(np.around(orig_pred, 2).astype(float))
        # meta_data['output']['original']['number_of_nodes'] = cell_graph.number_of_nodes()
        # meta_data['output']['original']['number_of_edges'] = cell_graph.number_of_edges()
        # meta_data['output']['original']['prediction'] = label_to_tumor_type[np.argmax(orig_pred)]

        # # 3-d explanation graph properties
        # meta_data['output']['explanation'] = {}
        # meta_data['output']['explanation']['number_of_nodes'] = explanation.number_of_nodes()
        # meta_data['output']['explanation']['number_of_edges'] = explanation.number_of_edges()
        # meta_data['output']['explanation']['logits'] = list(np.around(exp_pred, 2).astype(float))
        # meta_data['output']['explanation']['prediction'] = label_to_tumor_type[np.argmax(exp_pred)]
        # meta_data['output']['explanation']['node_importance'] = str(list(node_importance))
        # meta_data['output']['explanation']['centroids'] = str([list(centroid.cpu().numpy()) for centroid in graph_visualizer.centroid_cg])
        # meta_data['output']['explanation']['edges'] = str(list(graph_visualizer.edges_cg))

        # meta_data['output']['random'] = {'res': []}
        # 2. run random nuclei selector
        # for random_try_idx in range(config['explainer']['num_rand_tries']):

        #     # b. visualize the explanation graph
        #     graph_visualizer = GraphVisualization(save_path=args.out_path, show_centroid=True, show_edges=True)
        #     instance_map = data[-1][0]
        #     pruned_instance_map = np.zeros(instance_map.shape)
        #     for node_id in np.nonzero(node_idx):
        #         pruned_instance_map += instance_map * (instance_map == node_id.item() + 1)
        #     tmp_data = (explanation, data[1], [data[2][0] + '_explanation'], [pruned_instance_map])
        #     graph_visualizer(show_cg_flag, show_sp_flag, show_sp_map, tmp_data, 1, node_importance=None)

        #     # 3-c original graph properties
        #     meta_data['output']['original'] = {}
        #     meta_data['output']['original']['logits'] = list(np.around(orig_pred, 2).astype(float))
        #     meta_data['output']['original']['number_of_nodes'] = cell_graph.number_of_nodes()
        #     meta_data['output']['original']['number_of_edges'] = cell_graph.number_of_edges()
        #     meta_data['output']['original']['prediction'] = label_to_tumor_type[str(np.argmax(orig_pred))]

        #     # 3-d explanation graph properties
        #     meta_data['output']['explanation'] = {}
        #     meta_data['output']['explanation']['number_of_nodes'] = explanation.number_of_nodes()
        #     meta_data['output']['explanation']['number_of_edges'] = explanation.number_of_edges()
        #     meta_data['output']['explanation']['logits'] = list(np.around(exp_pred, 2).astype(float))
        #     meta_data['output']['explanation']['prediction'] = label_to_tumor_type[str(np.argmax(exp_pred))]
        #     meta_data['output']['explanation']['node_importance'] = str(list(node_importance))
        #     meta_data['output']['explanation']['centroids'] = str([list(centroid.cpu().numpy()) for centroid in graph_visualizer.centroid_cg])
        #     meta_data['output']['explanation']['edges'] = str(list(graph_visualizer.edges_cg))

        #     # 2. run random nuclei selector
        #     rand_adj, rand_feats, orig_pred, rand_pred = random_selector.run(
        #         graph=cell_graph,
        #         keep_prob=explanation.number_of_nodes() / cell_graph.number_of_nodes()
        #     )

        #     node_idx = (rand_feats.sum(dim=-1) != 0.).squeeze().cpu()
        #     rand_adj = rand_adj[node_idx, :]
        #     rand_adj = rand_adj[:, node_idx]
        #     rand_feats = rand_feats[node_idx, :]
        #     pruned_centroids = centroids[node_idx, :]
        #     random_graph = adj_to_networkx(rand_adj, rand_feats, threshold=config['explainer']['adj_thresh'], centroids=pruned_centroids)

        #     # b. visualize the random graph (only the 1st one)
        #     graph_visualizer = GraphVisualization(save_path=args.out_path, show_centroid=True, show_edges=True)
        #     instance_map = data[-1][0]
        #     pruned_instance_map = np.zeros(instance_map.shape)
        #     for node_id in np.nonzero(node_idx):
        #         pruned_instance_map += instance_map * (instance_map == node_id.item() + 1)
        #     tmp_data = (random_graph, data[1], [data[2][0] + '_random'], [pruned_instance_map])
        #     graph_visualizer(show_cg_flag, show_sp_flag, show_sp_map, tmp_data, 1)

        #     # 3-d random_graph graph properties
        #     rand_res = {
        #         'number_of_nodes': random_graph.number_of_nodes(),
        #         'number_of_edges': random_graph.number_of_edges(),
        #         'logits': list(np.around(rand_pred, 2).astype(float)),
        #         'prediction': label_to_tumor_type[str(np.argmax(rand_pred))],
        #         'centroids': str([list(centroid.cpu().numpy()) for centroid in graph_visualizer.centroid_cg]),
        #         'edges': str(list(graph_visualizer.edges_cg))
        #     }
        #     meta_data['output']['random']['res'].append(rand_res)

        # 3-e write to json
        # write_json(complete_path(args.out_path, data[-2][0] + '.json'), meta_data)

        # 4. aggregate all the information in a single user-friendly image
        # agg_and_plot_interpretation(meta_data, save_path=args.out_path, image_name=data[-2][0])

        # except:
        #     print('An error occured')


if __name__ == "__main__":
    main(args=parse_arguments())
