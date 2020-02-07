from skimage.future import graph
import h5py
import os
import argparse
import dgl
import torch
from dgl.data.utils import save_graphs
import numpy as np

from histocartography.utils.io import h5_to_tensor, get_device, check_for_dir
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN, CENTROID


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data_path',
        type=str,
        help='path to the h5 files data.',
        default='/dataT/pus/histocartography/Data/pascale/super_pixel_info/main_sp/prob_thr_0.8/',
        required=False
    )

    parser.add_argument(
        '-s',
        '--save_path',
        type=str,
        help='path for dgl files to be saved',
        default='../../dgl_graphs/prob_thr_0.8/',
        required=False
    )

    return parser.parse_args()


# if running in GPU
CUDA = torch.cuda.is_available()
device = get_device(CUDA)


def main(args):
    """

    Converts the RAG graph to DGL graph and saves it as .bin file in the save_path
    """
    dirs = [name for name in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, name))]
    h5_names = []
    for _dir in dirs:
        path = os.path.join(args.data_path, _dir)
        files = os.listdir(path)
        for file in files:
            if file.endswith('.h5'):
                h5_names.append(os.path.join(path, file))
                with h5py.File(os.path.join(path, file), 'r') as f:
                    # load SP map and centroids
                    d_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
                    centroid = h5_to_tensor(f['sp_centroids'], device).type(d_type)
                    sp_map = h5_to_tensor(f['sp_map'], device).type(d_type)
                    # get number of nodes
                    num_nodes = centroid.shape[0]
                    f.close()

                    # construct RAG
                    sp_map = sp_map.cpu()
                    g = graph.RAG(np.array(sp_map), connectivity=2)

                    # make dgl graph
                    dgl_g = dgl.DGLGraph()
                    dgl_g.add_nodes(num_nodes)
                    dgl_g.from_networkx(g)
                    dgl_g.ndata[GNN_NODE_FEAT_IN] = torch.zeros((num_nodes, 57)) #TODO: check features number : may change
                    dgl_g.ndata[CENTROID] = torch.zeros((num_nodes, centroid.shape[1]))

                    # save dgl graph in save_path + dir+ filename
                    check_for_dir(os.path.join(args.save_path, _dir))
                    dgl_file = os.path.splitext(file)[0] + '.bin'
                    save_graphs(os.path.join(args.save_path, _dir, dgl_file), [dgl_g])


if __name__ == "__main__":
    main(args=parse_arguments())

