# Preprocessing Pipeline

## How to Use
To use the preprocessing pipeline, you need to define a pipeline that connects the modules. For example a pipeline could be:
- Stain Normalization
- Superpixel Extraction
- Feature Extraction
- Graph Building

First a configuration needs to be defined, then the pipeline can be run directly.

### Configuration
These steps must be defined in a configuration `.yml` file with the following structure:

```yaml
inputs:
- input1
- input2
- ...
outputs:
- output1
- output2
- ...
stages:
  - module1:
      class: class_name1 # (str) class to import from histocartography.preprocessing.module1
      inputs:
      - input1
      outputs:
      - output1
      params:
        # Keyword arguments to histocartography.preprocessing.module1.class_name1.__init__
  - module2:
      class: class_name2 # (str) class to import from histocartography.preprocessing.module2
      inputs:
      - input2
      - output1
      outputs:
      - output1
      params:
        # Keyword arguments to histocartography.preprocessing.module2.class_name2.__init__
  - ...
```

An example of such a config can be found in `config.yml`.

### Running single datapoints
To run the pipeline for a single datapoint, use the following Python code:
```python
import yaml
from histocartography.preprocessing.pipeline import PipelineRunner
with open('PATH_TO_CONFIG', 'r') as file:
    config = yaml.load(file)
pipeline = PipelineRunner(output_path="PATH_TO_OUTPUT", **config)
pipeline.precompute()
output = pipeline.run(name="IDENTIFIER", input1=INPUT1, input2=INPUT2)
```

Make sure to use the same keyword arguments in the call of run as you defined in the config. Here we speficied the inputs to be input1 and input2, so we provide them at run time.

The outputs that are computed at a dictionary with keys as defined in the config and the values that were computed in the pipeline.

### Running a whole batch of datapoints
Typically the preprocessing needs to be applied to a whole colletion of inputs. Due to the lack of dependencies between datapoints, this can be done in a multiprocessed fashion. To run the pipeline like this, use the following Python code:
```python
import yaml
from histocartography.preprocessing.pipeline import BatchPipelineRunner
with open('PATH_TO_CONFIG', 'r') as file:
    config = yaml.load(file)
pipeline = BatchPipelineRunner(output_path="PATH_TO_OUTPUT", **config)
df = BUILD_DF()
output = pipeline.run(metadata=df, cores=4)
```

Note: the `BUILD_DF` function should build a `pandas.DataFrame` that has the following structure: the index corresponds to the unique datapoint identifier (e.g. a filename). Each column has the name as specified in the config under inputs, and values that correspond to the elements to be passed to the pipeline step with those inputs. Typically the dataframe consists of paths that are then passed to an io pipeline step that loads the resources.

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