"""Base explainer."""

from abc import abstractmethod
import numpy as np 
import dgl 
import torch 
from typing import Tuple
from mlflow.pytorch import load_model

from ..preprocessing.pipeline import PipelineStep
from ..utils.io import is_mlflow_url


class BaseExplainer(PipelineStep):
    """Base pipelines step"""

    def __init__(self, model_path, **kwargs) -> None:
        """Abstract class that define the base explainer. 

        Args:
            model_path (model_path): Model path to pre-trained model. Required. 
                                     The path can be local or an MLflow URL. 
        """
        super().__init__(**kwargs)
        self.model_path = model_path 

        # look for GPU
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")

        # load model 
        if is_mlflow_url(model_path):
            self.model = load_model(model_path,  map_location=torch.device('cpu'))
        else:
            self.model = torch.load(model_path)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model.zero_grad()

    @abstractmethod
    def process(self, graph: dgl.DGLGraph, label: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Explain a graph
        
        Args:
            graph (dgl.DGLGraph): Input graph to explain
            label (int): Label attached to the graph. Default to None. 
        """
