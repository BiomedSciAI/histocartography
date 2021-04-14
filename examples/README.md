# Examples

We present a set of small scripts that should help you understand the capabilities of the `histocartography` library. 
All the examples are self-contained and should directly work without the need to download any data or model. 
Naturally, the examples are limited and are not comprehensive, but should remain a good starting point to develop
your own application with your data. 

## Example 1: Stain Normalization with Vahadane et al. algorithm

Run the script as:
`python stain_normalization.py`

This script will apply stain normalization to a set of images using the well-know algorithm proposed by Vahadane et al.
A reference image is used to evaluate the transformation parameters, parameters that are further applied to each image. 

Note: The library also includes the Macenko stain normalization algortihm. 

**References:**

- [Structure-Preserving Color Normalization and SparseStain Separation for Histological Images.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7460968) Vahadane et al., IEEE Transactions on Medical Imaging, 2016.
- [A method for normalizing histology slides for quantitative analysis.](https://ieeexplore.ieee.org/document/5193250) Macenko et al., IEEE International Symposium on Biomedical Imaging, 2009.

## Example 2: Cell Graph (CG) generation

Run the script as:
`python cell_graph_generation.py`

This example will guide you to generate a Cell Graph (CG) from an H&E stained histology image. Example images are taken from the BRACS dataset, a large cohort of breast cancer tumor regions-of-interest. The CG generation begins with extracting nuclei, i.e. nodes, from a stained normalized image using the HoverNet model. Then, node features are extracted using a ResNet34 model pre-trained on ImageNet. Finally, the graph is built by connecting nuclei, i.e., nodes, to form a kNN graph. In this example, we set k to 5 and remove edges longer than 50 pixels.
**References:**

- [BRACS: BReAst Carcinoma Subtyping.](https://www.bracs.icar.cnr.it/) 2021.
- [Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images.](https://arxiv.org/pdf/1812.06499.pdf) Graham et al., MEDIA, 2019.
- [Hierarchical Graph Representations in Digital Pathology.](https://arxiv.org/pdf/2102.11057.pdf) Pati et al., 	arXiv:2102.11057, 2021.
- [Quantifying Explainers of Graph Neural Networks in Computational Pathology.](https://arxiv.org/pdf/2011.12646.pdf) Jaume et al., CVPR, 2021.

## Example 3: Tissue Graph (TG) generation

Run the script as:
`python tissue_graph_generation.py`

Following the Cell Graph generation example, this script will show you how to build a Tissue Graph (TG) from a stained normalized image. First, we start by extracting tissue regions in an unsupersived fashion using the SLIC superpixel extraction algorithm followed by  color merging-based superpixel merging. Then, deep features are extracted to represent the superpixels. Finally, the graph is built by connecting adjacent regions to each other resulting in a Region Adjacency Graph (RAG). 

## Example 4: Feature cube extraction

Run the script as:
`python feature_cube_extraction.py`

This example will extract features on each patch of a set of H&E images. The resulting output is a feature cube 
where the x and y dimensions are the number of patches along the x and y, respectively and, where the number of channels 
is the dimensionality of the patch embedding before the classification layer. 

In this script, the patch size is set to 224 with no stride. The model is a ResNet34 network pre-trained on ImageNet. 

Note: All the `torchvision` pretrained architectures will work. 

## Example 5: Cell Graph explainer

Run the script as:
`python cell_graph_explainer.py`

This example allows you to generate an explanation, i.e., node-level importance scores, using the GraphGradCAM algortihm, a post-hoc explanability technique. The library also provides other graph-based explaining techniques, i.e., GNNExplainer, GraphLRP or GraphGradCAM++. 

**References:**

- [Grad-CAM : Visual Explanations from Deep Networks.](https://arxiv.org/pdf/1610.02391.pdf) Selvaraju et al., ICCV, 2017. 
- [Explainability methods  for graph  convolutional  neu-ral  networks.](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.pdf) Pope et al., CVPR, 2019. 
- [Quantifying Explainers of Graph Neural Networks in Computational Pathology.](https://arxiv.org/pdf/2011.12646.pdf) Jaume et al., CVPR, 2021.
