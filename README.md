<p align="center">
  <img src="https://raw.githubusercontent.com/histocartography/histocartography/main/docs/_static/logo_large.png" height="200">
</p>

[![Build Status](https://travis-ci.com/histocartography/histocartography.svg?branch=main)](https://travis-ci.com/histocartography/histocartography)
[![codecov](https://codecov.io/gh/histocartography/histocartography/branch/main/graph/badge.svg?token=OILOGEBP0Q)](https://codecov.io/gh/histocartography/histocartography)
[![PyPI version](https://badge.fury.io/py/histocartography.svg)](https://badge.fury.io/py/histocartography)
![GitHub](https://img.shields.io/github/license/histocartography/histocartography)
[![Downloads](https://pepy.tech/badge/histocartography)](https://pepy.tech/project/histocartography)

**[Documentation](https://histocartography.github.io/histocartography/)**
| **[Paper](https://arxiv.org/pdf/2107.10073.pdf)** 

**Welcome to the `histocartography` repository!** `histocartography` is a python-based library designed to facilitate the development of graph-based computational pathology pipelines. The library includes plug-and-play modules to perform,
- standard histology image pre-processing (e.g., *stain normalization*, *nuclei detection*, *tissue detection*)
- entity-graph representation building (e.g. *cell graph*, *tissue graph*, *hierarchical graph*)
- modeling Graph Neural Networks (e.g. *GIN*, *PNA*)
- feature attribution based graph interpretability techniques (e.g. *GraphGradCAM*, *GraphGradCAM++*, *GNNExplainer*)
- visualization tools 

All the functionalities are grouped under a user-friendly API. 

If you encounter any issue or have questions regarding the library, feel free to [open a GitHub issue](add_link). We'll do our best to address it. 

# Installation 

## PyPI installer (recommended)

`pip install histocartography`

## Development setup 

- Clone the repo:

```
git clone https://github.com/histocartography/histocartography.git && cd histocartography
```

- Create a conda environment:

```
conda env create -f environment.yml
```

- Activate it:

```
conda activate histocartography
```

- Add `histocartography` to your python path:

```
export PYTHONPATH="<PATH>/histocartography:$PYTHONPATH"
```

## Tests

To ensure proper installation, run unit tests as:

```sh 
python -m unittest discover -s test -p "test_*" -v
```

Running tests on cpu can take up to 20mn. 

# Using histocartography 

The `histocartography` library provides a set of helpers grouped in different modules, namely `preprocessing`, `ml`, `visualization` and `interpretability`.  

For instance, in `histocartography.preprocessing`, building a cell-graph from an H&E image is as simple as:

```
>> from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder
>> 
>> nuclei_detector = NucleiExtractor()
>> feature_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72)
>> knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)
>>
>> image = np.array(Image.open('docs/_static/283_dcis_4.png'))
>> nuclei_map, _ = nuclei_detector.process(image)
>> features = feature_extractor.process(image, nuclei_map)
>> cell_graph = knn_graph_builder.process(nuclei_map, features)
```

The output can be then visualized with:

```
>> from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization

>> visualizer = OverlayGraphVisualization(
...     instance_visualizer=InstanceImageVisualization(
...         instance_style="filled+outline"
...     )
... )
>> viz_cg = visualizer.process(
...     canvas=image,
...     graph=cell_graph,
...     instance_map=nuclei_map
... )
>> viz_cg.show()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/histocartography/histocartography/main/docs/_static/283_dcis_4_cg.png"  height="400">
</p>

A list of examples to discover the capabilities of the `histocartography` library is provided in `examples`. The examples will show you how to perform:

- **stain normalization** with Vahadane or Macenko algorithm
- **cell graph generation** to transform an H&E image into a graph-based representation where nodes encode nuclei and edges nuclei-nuclei interactions. It includes: nuclei detection based on HoverNet pretrained on PanNuke dataset, deep feature extraction and kNN graph building. 
- **tissue graph generation** to transform an H&E image into a graph-based representation where nodes encode tissue regions and edges tissue-to-tissue interactions. It includes: tissue detection based on superpixels, deep feature extraction and RAG graph building. 
- **feature cube extraction** to extract deep representations of individual patches depicting the image
- **cell graph explainer** to generate an explanation to highlight salient nodes. It includes inference on a pretrained CG-GNN model followed by GraphGradCAM explainer. 

A tutorial with detailed descriptions and visualizations of some of the main functionalities is provided [here](https://github.com/maragraziani/interpretAI_DigiPath/blob/feature/handson2%2Fpus/hands-on-session-2/hands-on-session-2.ipynb) as a notebook. 

# External Ressources 

## Learn more about GNNs 

- We have prepared a gentle introduction to Graph Neural Networks. In this tutorial, you can find [slides](https://github.com/guillaumejaume/tuto-dl-on-graphs/blob/main/slides/ml-on-graphs-tutorial.pptx), [notebooks](https://github.com/guillaumejaume/tuto-dl-on-graphs/tree/main/notebooks) and a set of [reference papers](https://github.com/guillaumejaume/tuto-dl-on-graphs).
- For those of you interested in exploring Graph Neural Networks in depth, please refer to [this content](https://github.com/guillaumejaume/graph-neural-networks-roadmap) or [this one](https://github.com/thunlp/GNNPapers).


## Papers already using this library

- Hierarchical Graph Representations for Digital Pathology, Pati et al., preprint, 2021. [[pdf]](https://arxiv.org/pdf/2102.11057.pdf) [[code]]() 
- Quantifying Explainers of Graph Neural Networks in Computational Pathology,  Jaume et al., CVPR, 2021. [[pdf]](https://arxiv.org/pdf/2011.12646.pdf) [[code]](https://github.com/histocartography/patho-quant-explainer) 
- Learning Whole-Slide Segmentation from Inexact and Incomplete Labels using Tissue Graphs, Anklin et al., preprint, 2021. [[pdf]](https://arxiv.org/pdf/2103.03129.pdf) [[code]]() 

If you use this library, please consider citing:

```
@inproceedings{jaume2021,
    title = {HistoCartography: A Toolkit for Graph Analytics in Digital Pathology},
    author = {Guillaume Jaume, Pushpak Pati, Valentin Anklin, Antonio Foncubierta, Maria Gabrani},
    booktitle = {https://arxiv.org/pdf/2107.10073},
    year = {2021}
} 

@inproceedings{pati2021,
    title = {Hierarchical Graph Representations for Digital Pathology},
    author = {Pushpak Pati, Guillaume Jaume, Antonio Foncubierta, Florinda Feroce, Anna Maria Anniciello, Giosuè Scognamiglio, Nadia Brancati, Maryse Fiche, Estelle Dubruc, Daniel Riccio, Maurizio Di Bonito, Giuseppe De Pietro, Gerardo Botti, Jean-Philippe Thiran, Maria Frucci, Orcun Goksel, Maria Gabrani},
    booktitle = {Medical Image Analysis (MEDIA)},
    year = {2021}
} 
```
