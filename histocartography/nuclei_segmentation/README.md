# Nuclei Segmentation using HoVerNet

HoverNet: A multiple branch network that performs nuclear instance segmentation and classification within a single network.

## Inference

To generate the network predictions on the dataset, run `infer.py`.

### Modifying parameters for running inference

To modify the parameters for the network, make changes to `config.py`.
* To set if classification of nuclei is done or not, set type_classification = False(if only instance segmentation) and True if classification
* Set path to checkpoint (`inf_model_path`)
* Set path to images (`inf_data_dir`)
* Set output directory (`inf_output_dir`)

## Post-processing to json files after inference

After the results of the network is obtained from `infer.py` (will be located in the inf_output_dir in .mat format).
Run `process.py` to obtain the images with the instances overlaid as well as the json files.

### JSON file structure

If classification(type_classification set to True):

1. 'detected_instance_map' : The instance map for the predicted instances
2. 'detected_type_map' : Map showing the type classification of the instances
3. 'instance_types' : The classification type (1- Miscellaneous, 2- Inflammatory, 3- Epithelial 4- Spindle-shaped. 0 stands for background)
4. 'instance_centroid_location' : Centroid location of each instance (X and Y coordinates) : follows (x- horizontal to right, and y- vertical to bottom)
5. 'image_dimension' : Dimension of the patch [H, W, C]

If just instance segmentation(type_classification set to False):

1. 'detected_instance_map' : The instance map for the predicted instances
2. 'instance_centroid_locations' : Centroid location of each instance (X and Y coordinates) : follows (x- horizontal to right, and y- vertical to bottom)
3. 'image_dimension' : Dimension of the patch [H, W , C]


