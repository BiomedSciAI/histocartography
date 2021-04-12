<p align="center">
  <img src="https://ibm.box.com/shared/static/hzutz91ez9ukpzbxk69iy5snk8ijs8p9.png" height="200">
</p>

[![Build Status](https://travis.ibm.com/DigitalPathologyZRL/histocartography.svg?token=8FJcyLKb64p4ANuB6hHj&branch=cleanup/stable)](https://travis.ibm.com/DigitalPathologyZRL/histocartography)

High-level description of the library. 

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
python3 -m unittest discover -v
```

- Activate it:

```
conda activate histocartography
```

# Using histocartography 

Example 1:

Example 2:

# Papers already using this library

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
