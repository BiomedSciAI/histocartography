# histocartography
[![Build Status](https://travis.ibm.com/DigitalPathologyZRL/histocartography.svg?token=8FJcyLKb64p4ANuB6hHj&branch=master)](https://travis.ibm.com/DigitalPathologyZRL/histocartography)

## Suggested setup for development

- Create the conda environment:

```
conda env create -f environment.yml
```

- Activate it:

```
conda activate histocartography
```

## Testing and Continuous Integration
Add these environment variables to have access to the [Object Storage](http://data.digital-pathology.zc2.ibm.com:9000)
```
export AWS_ACCESS_KEY_ID=" "
export AWS_SECRET_ACCESS_KEY=" "
```
### Training

There is a sample executable script in the `bin/` folder that can be used 
for training Histocartography machine learning pipelines.

### Unit Tests

Unit tests should be created in the `test/` folder.
Run the tests with: 

```sh 
python3 -m unittest discover -v
```

## Docker support

The repo contains a `Dockerfile` that builds an image containing the python package.
