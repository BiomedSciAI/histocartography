# histocratography
[![Build Status](https://travis.ibm.com/DigitalPathologyZRL/histocartography.svg?token=8FJcyLKb64p4ANuB6hHj&branch=master)](https://travis.ibm.com/DigitalPathologyZRL/histocartography)

Histocartography python package. It installs `histocartography` using `pip`.

## Suggested setup for development

Create a `virtualenv`:

```sh
python3 -m venv venv
```

Activate it:

```sh
source venv/bin/activate
```

Install the package as editable and any dependencies:

```sh
pip3 install -e .
```

## Testing and Continuous Integration

### Training

There is a sample executable script in the `bin/` folder that can be used 
for training Histocartography.

### Unit Tests

Unit tests should be created in the `test/` folder.
Run the tests with: 

```sh 
python3 -m unittest discover -v
```

## Docker support

The repo contains a `Dockerfile` that builds an image containing the python package.
