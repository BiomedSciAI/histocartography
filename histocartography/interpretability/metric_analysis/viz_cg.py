import os
import json 
import numpy as np
from dgl.data.utils import load_graphs
from PIL import Image

from histocartography.utils.visualization import GraphVisualization, overlay_mask


PATH = '/Users/gja/Documents/PhD/histocartography/data/283_dcis_4'

GRAPH_FNAME = '283_dcis_4.bin'
OUT_FNAME = '283_dcis_4_explanation.json'
IMAGE_FNAME = '283_dcis_4.png'

# 1. open cell graph
g, _ = load_graphs(os.path.join(PATH, GRAPH_FNAME))
g = g[0]
print('Graph is:', g)

# 2. open json 
with open(os.path.join(PATH, OUT_FNAME)) as f:
    data = json.load(f)
    output = data['output']

label_index = output['label_index']
explanation = output['explanation']['1']        # str(self.args.p)
logits = np.asarray(explanation['logits'])
node_importance = np.asarray(explanation['node_importance'])
node_centroid = np.asarray(explanation['centroid'])

# 3. load image 
image = Image.open(os.path.join(PATH, IMAGE_FNAME))

# 4. overlay graph edges + node with color coding based on node importance 
visualizer = GraphVisualization(show_edges=True, save=True, save_path='../../../data/283_dcis_4/output/')
explanation_as_image = visualizer(
    show_cg=True,
    show_sg=False,
    data=[g, image, '283_dcis_4'],
    node_importance=node_importance
)