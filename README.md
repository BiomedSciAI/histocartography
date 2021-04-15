from histocartography.visualization import InstanceImageVisualization
from histocartography.preprocessing import NucleiExtractor
<p align = "center" >
  <img src = "https://ibm.box.com/shared/static/bgv4hk068bv26s8q6pasoxi401tt6b55.png" height = "200" >
< / p >

[![Build Status](https: // travis.ibm.com / DigitalPathologyZRL / histocartography.svg?token=8FJcyLKb64p4ANuB6hHj & branch=cleanup / stable)](https: // travis.ibm.com / DigitalPathologyZRL / histocartography)

High - level description of the library.

# Installation

# PyPI installer (recommended)

`pip install histocartography`

# Development setup

- Clone the repo:

```
git clone < ADD - PUBLIC - URL > & & cd histocartography
```

- Create a conda environment:

```
conda env create - f environment.yml
```

- Activate it:

```
conda activate histocartography
```

- Add `histocartography` to your python path:

```
export PYTHONPATH = "<PATH>/histocartography:$PYTHONPATH"
```

# Tests

To ensure proper installation, run unit tests as:

```sh
python - m unittest discover - s test - p "test_*" - v
```

# Using histocartography

The `histocartography` library provides a set of helpers grouped in different modules, namely `preprocessing`, `visualization`, `ml` and `interpretability`.

For instance, in `histocartography`, detecting nuclei in an H & E image is as simple as:

```

detector = NucleiExtractor()
image = np.array(Image.open('images/283_dcis_4.png'))
instance_map, _ = detector.process(image)
```

The output can be then visualized with:

```

visualizer = InstanceImageVisualization()
canvas = visualizer.process(image, instance_map=instance_map)
canvas.show()
```

<p align = "center" >
  <img src = "" height = "200" >
< / p >

A list of examples to discover the capabilities of the `histocartography` library is provided in `examples`.

A tutorial with detailed descriptions and visualizations of some of the main functionalities is provided[here](https: // github.com / maragraziani / interpretAI_DigiPath / blob / feature / handson2 % 2Fpus / hands - on - session - 2 / hands - on - session - 2.ipynb) as a notebook.

# Papers already using this library

- [Hierarchical Graph Representations for Digital Pathology](https: // arxiv.org / pdf / 2102.11057.pdf]), Pati et al., preprint, 2021.
- [Quantifying Explainers of Graph Neural Networks in Computational Pathology](https: // arxiv.org / pdf / 2011.12646.pdf), Jaume et al., CVPR, 2021.

If you use this library, please consider citing:

```
@inproceedings{pati2021,
    title = {Hierarchical Graph Representations for Digital Pathology},
    author = {Pushpak Pati, Guillaume Jaume, Antonio Foncubierta, Florinda Feroce, Anna Maria Anniciello, Giosuè Scognamiglio, Nadia Brancati, Maryse Fiche, Estelle Dubruc, Daniel Riccio, Maurizio Di Bonito, Giuseppe De Pietro, Gerardo Botti, Jean - Philippe Thiran, Maria Frucci, Orcun Goksel, Maria Gabrani},
    booktitle = {https: // arxiv.org / pdf / 2102.11057},
    year = {2021}
}
```
