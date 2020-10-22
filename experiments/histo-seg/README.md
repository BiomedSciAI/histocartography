# Weakly Supervised Semantic Segmentation
This section of the code contains all code necessary to run the weakly supervised semantic segmentation (WSSS) pipeline. It consists of a preprocessing step that is run only once and a training procedure that is run many times and a post-processing step that is run after the training has completed.

## Installation
For convenience and reproducibility, we include a `environment.yml` that contains the exact library versions used when conducting experiments

To install run `conda create -f environment.yml`.

### Installing on the IBM Cluster
For this first add the IBM-AI conda channel:
```
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
```
Then install the dependencies as explained above. You might want to comment out the DGL dependency (if it is still not available) and install that from source (if there are problems, ask Guillaume (gja@zurich.ibm.com)):
```
git clone --recurse https://github.com/dmlc/dgl.git
cd dgl
mkdir build
cd build
cmake ..
make
```

## Data Download
### ETH Dataset
- Download the data at: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP
- Copy the data to the folder of your choosing
- Add the location of the data to the `eth.py` file as a `BASE_PATH`. All other scripts take the location from that file.
- Run the cleanup script with `python eth.py`.

## Preprocessing
The preprocessing pipeline consits of multiple steps that are computed sequentially. In general it consists of the following steps:
- Stain Normalization
- Superpixel Extraction
- Feature Extraction
- Graph Building
To run the preprocessing, first it is needed to define a configuration that describes the steps and parameters:

### Configuration
The configuration is defined in a `.yml` file with the following structure:

```
preprocess:
  stages:
    stain_normalizer:
      class: # (str) Class to use for stain normalization. Must be descendant of StainNormalizer and defined in stain_normalizers.py
      params:
        # Keyword arguments of the StainNormalizer
    superpixel_extractor:
      class: # (str) Class to use for superpixel extraction. Must be descendant of SuperpixelExtractor and defined in superpixel.py
      params:
        # Keyword arguments of the SuperpixelExtractor
    feature_extractor:
      class: # (str) Class to use for feature extraction. Must be descendant of FeatureExtractor and defined in feature_extraction.py
      params:
        # Keyword arguments of the FeatureExtractor
    graph_builder:
      class: # (str) Class to use for graph building. Must be descendant of BaseGraphBuilder and defined in graph_builders.py
      params:
        # Keyword arguments of the class BaseGraphBuilder:
  params:
    # Here go all the parameters of the preprocessing runner itself.
    # Most notable the number of cores to use to parallelize the preprocessing
```

An example of such a config can be found in `default.yml`.

### Running single preprocessing run locally
To run a single preprocessing run, you need to call: `python preprocess.py --config <PATH_TO_CONFIG>`

### Running single preprocessing run on the cluster
- Push the code to the cluster (e.g. using `git` or `rsync`) and make sure the data is accessible from the `BASE_PATH` defined in the dataset files (e.g. `eth.py`).
- Submit a standard job: `bsub < preprocessing.lsf`
- You can modify all parameters queues in that file if needed. Using just this command should run the currently best preprocessing pipeline.

### Run a whole batch of experiments
For this use case, there exists a Python script to generate experiment configuration and corresponding LSF files. To run an existsing experiment config (consisting of multiple jobs) you can directly run `python generate_experiments.py`.

If you wish to include your own experiment, you can create a new function in `generate_experiments.py`. As a basis, you can look at the implementation of `generate_performance_test` in the same file.

After the experiment configurations has been generated, you can submit them on the job with the command: `./submitter.sh <ABSOLUTE_PATH_TO_CONFIGS>` which will schedule all the jobs (.lsf files) that are located in `<ABSOLUTE_PATH_TO_CONFIGS>`. This needs to be an absolute path, as it otherwise will not find the correct configurations.

### Preprocessing structure
To generate this structure use this command: `tree -v --charset utf-8 -I '*.egg-info|__pycache__'`

```
.
├── README.md                   # This file
├── constants.py                # Contains all constants that are the same for the whole project
├── default.yml                 # Default experiment configuration
├── environment.yml             # Needed to create the environment
├── eth.py                      # Contains all constants and functions that are specific to the ETH dataset
├── feature_extraction.py       # Contains all feature extractors
├── generate_experiments.py     # Script to generate experiment configurations (e.g. for grid search)
├── graph_builders.py           # Contains all graph builders
├── preprocess.py               # Main executable for the preprocessing pipeline
├── preprocessing.lsf           # Default configuration submitting the preprocessing pipeline
├── stain_normalizers.py        # Contains all stain normalizers
├── submitter.sh                # Script to submit a whole directory of jobs generated by generate_experiments.py
├── superpixel.py               # Contains all superpixel extractors
└── utils.py                    # Contains all utility functions that are useful across modules
```

## Training GNN
TODO: Not implemented yet

## Post-Processing Segmentation
TODO: Not implemented yet\
