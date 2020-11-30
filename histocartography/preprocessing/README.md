# Preprocessing Pipeline

## How to Use
To use the preprocessing pipeline, you need to define a main script that connects the modules. For example a pipeline could be:
- Stain Normalization
- Superpixel Extraction
- Feature Extraction
- Graph Building
An example of such a pipeline can be found in `experiments/histo-seg/preprocess.py`.

Parameters for the steps are passed as keyword arguments read from a configuration `.yml` file needs to be defined. For example it can look as follows:

```yaml
preprocess:
  stages:
    stain_normalizer:
      class: # (str) Class to use for stain normalization.
             # Must be descendant of StainNormalizer and defined in stain_normalizers.py
      params:
        # Keyword arguments of the StainNormalizer
    superpixel_extractor:
      class: # (str) Class to use for superpixel extraction.
             # Must be descendant of SuperpixelExtractor and defined in superpixel.py
      params:
        # Keyword arguments of the SuperpixelExtractor
    feature_extractor:
      class: # (str) Class to use for feature extraction.
             # Must be descendant of FeatureExtractor and defined in feature_extraction.py
      params:
        # Keyword arguments of the FeatureExtractor
    graph_builder:
      class: # (str) Class to use for graph building.
             # Must be descendant of BaseGraphBuilder and defined in graph_builders.py
      params:
        # Keyword arguments of the class BaseGraphBuilder:
  params:
    # Here go all the parameters of the preprocessing runner itself.
    # Most notable the number of cores to use to parallelize the preprocessing
```

An example of such a config can be found in `experiments/histo-seg/default.yml`. 

## Preprocessing structure
To generate this structure use this command: `tree -v --charset utf-8 -I '*.egg-info|__pycache__'`

```
├── README.md                   # This file
├── constants.py                # Contains all constants that are the same for the whole project
├── feature_extraction.py       # Contains all feature extractors
├── graph_builders.py           # Contains all graph builders
├── stain_normalizers.py        # Contains all stain normalizers
├── superpixel.py               # Contains all superpixel extractors
└── utils.py                    # Contains all utility functions that are useful across modules
```