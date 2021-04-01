"""Base explainer."""

from abc import abstractmethod
from typing import Optional, Tuple
import dgl
import numpy as np
import torch
import os 
from mlflow.pytorch import load_model

from ..pipeline import PipelineStep


class BaseExplainer(PipelineStep):
    """Base pipelines step"""

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        **kwargs
    ) -> None:
        """Abstract class that defines an explainer.

        Args:
            model_path (Optional[str], optional): Model path to pre-trained model. The path can be local or an MLflow URL. Defaults to None.
            model (Optional[torch.nn.Module], optional): PyTorch model to use. Defaults to None.
        """

        super().__init__(**kwargs)

        # look for GPU
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")

        # set model
        self.model = model
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model.zero_grad()

    @abstractmethod
    def process(
        self, graph: dgl.DGLGraph, class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Explain a graph

        Args:
            graph (dgl.DGLGraph): Input graph to explain
            class_idx (int): Class to explain. If None, use the winning class. Default to None.
        """


