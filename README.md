<p align="center">
  <img src="https://ibm.box.com/shared/static/568egs22ggn4gj080ys3m3dzd6fziga9.png" height="200">
</p>

[![Build Status](https://travis.ibm.com/DigitalPathologyZRL/histocartography.svg?token=8FJcyLKb64p4ANuB6hHj&branch=cleanup/stable)](https://travis.ibm.com/DigitalPathologyZRL/histocartography)
[![Build Status](https://travis.ibm.com/DigitalPathologyZRL/histocartography.svg?token=8FJcyLKb64p4ANuB6hHj&branch=cleanup/stable)](https://travis.ibm.com/DigitalPathologyZRL/histocartography)
[![Build Status](https://travis.ibm.com/DigitalPathologyZRL/histocartography.svg?token=8FJcyLKb64p4ANuB6hHj&branch=cleanup/stable)](https://travis.ibm.com/DigitalPathologyZRL/histocartography)
[![Build Status](https://travis.ibm.com/DigitalPathologyZRL/histocartography.svg?token=8FJcyLKb64p4ANuB6hHj&branch=cleanup/stable)](https://travis.ibm.com/DigitalPathologyZRL/histocartography)

**[Documentation](https://<documentation>)**
| **[Paper](https://arxiv.org/pdf/2102.11057.pdf)** 

Welcome to the `histocartography` repository! `histocartography` is a python-based library designed to ease the development of computational pathology pipelines. The library offers plug-and-play modules to address a number of common tasks in digital pathology, e.g., stain normalization, nuclei detection, tissue detection. Even more, `histocartography` includes a set helpers to work with graph-based representation of histology images. Features range from cell graph generation of an H&E image to Graph Neural Network-based modeling. All the features are grouped under a user-friendly API. 

If you encounter any issue or have questions regarding the library, feel free to [open a GitHub issue](add_link), we'll do our best to address it. 

# Installation 

## PyPI installer (recommended)

`pip install histocartography`

## Development setup 

- Clone the repo:

```
git clone <ADD-PUBLIC-URL> && cd histocartography
```

- Create a conda environment:

```
conda env create -f environment.yml
```

- Activate it:

```
conda activate histocartography
```

- Add `histocartography` to your python path:

```
export PYTHONPATH="<PATH>/histocartography:$PYTHONPATH"
```

## Tests

To ensure proper installation, run unit tests as:

```sh 
python -m unittest discover -s test -p "test_*" -v
```

# Using histocartography 

The `histocartography` library provides a set of helpers grouped in different modules, namely `preprocessing`, `visualization`, `ml` and `interpretability`.  

For instance, in `histocartography`, detecting nuclei in an H&E image is as simple as:

```
>> from histocartography.preprocessing import NucleiExtractor
>> 
>> detector = NucleiExtractor()
>> image = np.array(Image.open('images/283_dcis_4.png'))
>> instance_map, _ = detector.process(image)
```

The output can be then visualized with:

```
>> from histocartography.visualization import InstanceImageVisualization
>> 
>> visualizer = InstanceImageVisualization()
>> canvas = visualizer.process(image, instance_map=instance_map)
>> canvas.show()
```

<p align="center">
  <img src="" height="200">
</p>

A list of examples to discover the capabilities of the `histocartography` library is provided in `examples`. 

A tutorial with detailed descriptions and visualizations of some of the main functionalities is provided [here](https://github.com/maragraziani/interpretAI_DigiPath/blob/feature/handson2%2Fpus/hands-on-session-2/hands-on-session-2.ipynb) as a notebook. 

# External Ressources 

## Learn more about GNNs 

- We have prepared a gentle introduction to Graph Neural Networks. In this tutorial, you can find [slides](https://github.com/guillaumejaume/tuto-dl-on-graphs/blob/main/slides/ml-on-graphs-tutorial.pptx), [notebooks](https://github.com/guillaumejaume/tuto-dl-on-graphs/tree/main/notebooks) and a set of [reference papers](https://github.com/guillaumejaume/tuto-dl-on-graphs).
- For those of you interested in exploring Graph Neural Networks in depth, please refer to [this content](https://github.com/guillaumejaume/graph-neural-networks-roadmap) or [this one](https://github.com/thunlp/GNNPapers).


## Papers already using this library

- [Hierarchical Graph Representations for Digital Pathology](https://arxiv.org/pdf/2102.11057.pdf]), Pati et al., preprint, 2021.
- [Quantifying Explainers of Graph Neural Networks in Computational Pathology](https://arxiv.org/pdf/2011.12646.pdf),  Jaume et al., CVPR, 2021.

If you use this library, please consider citing:

```
@inproceedings{pati2021,
    title = {Hierarchical Graph Representations for Digital Pathology},
    author = {Pushpak Pati, Guillaume Jaume, Antonio Foncubierta, Florinda Feroce, Anna Maria Anniciello, Giosuè Scognamiglio, Nadia Brancati, Maryse Fiche, Estelle Dubruc, Daniel Riccio, Maurizio Di Bonito, Giuseppe De Pietro, Gerardo Botti, Jean-Philippe Thiran, Maria Frucci, Orcun Goksel, Maria Gabrani},
    booktitle = {https://arxiv.org/pdf/2102.11057},
    year = {2021}
} 
```
