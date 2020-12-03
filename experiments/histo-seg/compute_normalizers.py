import pandas as pd
import torch
from tqdm import tqdm

from constants import GNN_NODE_FEAT_IN
from dataset import GraphClassificationDataset
from eth import ANNOTATIONS_DF, IMAGES_DF, PREPROCESS_PATH
from utils import merge_metadata

normalizer_path = PREPROCESS_PATH / "outputs" / "normalizers"
if not normalizer_path.exists():
    normalizer_path.mkdir()

for link in (PREPROCESS_PATH / "outputs").iterdir():
    if link.name == ".DS_Store" or link.name == normalizer_path.name:
        continue
    df = merge_metadata(
        pd.read_pickle(IMAGES_DF),
        pd.read_pickle(ANNOTATIONS_DF),
        graph_directory=link,
        add_image_sizes=True,
    )
    graph_dataset = GraphClassificationDataset(
        metadata=df, patch_size=None, centroid_features="no"
    )
    all_features = list()
    for i in tqdm(range(len(graph_dataset)), desc=f"{link.name}"):
        all_features.append(graph_dataset[i][0].ndata[GNN_NODE_FEAT_IN].mean(axis=0))
    all_features = torch.cat(all_features)
    torch.save(all_features.mean(axis=0), normalizer_path / f"mean_{link.name}.pth")
    torch.save(all_features.std(axis=0), normalizer_path / f"std_{link.name}.pth")
