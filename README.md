# histocartography

Histocartography python package. It installs `histocartography` using `pip`.

## Required libraries:
- OpenSlide >= 3.4.1 [Source](https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz). Version 3.4.0 DOESN'T READ TIFFs properly.
- OpenCV >= 2.4

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


There is a sample executable script in the `experiments/example/` folder that can be used 
for training Histocartography machine learning pipelines.
