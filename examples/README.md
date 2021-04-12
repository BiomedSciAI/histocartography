# Examples

We present a set of small scripts that should help you understand the capabilities of the `histocartography` library. 
All the examples are self-contained and should directly work without the need to download any data nor model. 
Naturally, the examples are limited and are not comprehensive, but should remain a good starting point to develop
your own application with your data. 

## Example 1: Stain Normalization with Vahadane et al. algorithm

Run the script as:
`python stain_normalization.py`

This script will apply stain normalization to a set of images using the well-know algorithm proposed by Vahadane et al.
A reference image is used to evaluate the transformation parameters, parameters that are further applied to each image. 

Note: The library also includes the Macenko stain normalization algortihm. 

## Example 2: Cell Graph (CG) generation

Run the script as:
`python cell_graph_generation.py`


## Example 3: Tissue Graph (TG) generation

Run the script as:
`python tissue_graph_generation.py`


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
