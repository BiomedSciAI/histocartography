.. histocartography documentation master file, created by
   sphinx-quickstart on Mon Apr 19 12:15:21 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to histocartography's documentation!
============================================

`histocartography` is a python-based library designed to facilitate the development of graph-based computational pathology pipelines. The library includes plug-and-play modules to perform: 

* standard histology image pre-processing (e.g., *stain normalization*, *nuclei detection*, *tissue detection*)
* entity-graph representation building (e.g. *cell graph*, *tissue graph*, *hierarchical graph*)
* modeling Graph Neural Networks (e.g. *GIN*, *PNA*)
* feature attribution based graph interpretability techniques (e.g. *GraphGradCAM*, *GraphGradCAM++*, *GNNExplainer*)
* visualization tools 

All the functionalities are grouped under a user-friendly API. 


.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   api/histocartography

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`



Reference
---------
If you use `histocartography` in your projects, please cite the following:

.. code-block:: python

    @inproceedings{pati2021,
        title = {Hierarchical Graph Representations for Digital Pathology},
        author = {Pushpak Pati, Guillaume Jaume, Antonio Foncubierta, Florinda Feroce, Anna Maria Anniciello, Giosuè Scognamiglio, Nadia Brancati, Maryse Fiche, Estelle Dubruc, Daniel Riccio, Maurizio Di Bonito, Giuseppe De Pietro, Gerardo Botti, Jean-Philippe Thiran, Maria Frucci, Orcun Goksel, Maria Gabrani},
        booktitle = {https://arxiv.org/pdf/2102.11057},
        year = {2021}
    } 
