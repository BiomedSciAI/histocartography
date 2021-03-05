"""Base explainer."""

from abc import abstractmethod
from typing import Optional, Tuple

import dgl
import numpy as np
import torch
from mlflow.pytorch import load_model

from ..preprocessing.pipeline import PipelineStep
from ..utils.io import is_mlflow_url


class BaseExplainer(PipelineStep):
    """Base pipelines step"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        **kwargs
    ) -> None:
        """Abstract class that define the base explainer.

        Args:
            model_path (Optional[str], optional): Model path to pre-trained model. The path can be local or an MLflow URL. Defaults to None.
            model (Optional[torch.nn.Module], optional): PyTorch model to use. Defaults to None.
        """
        assert (
            model_path is not None or model is not None
        ), "Either a model_path or a model need to be provided."
        super().__init__(**kwargs)
        self.model_path = model_path

        # look for GPU
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")

        # load model

        if model_path is None:
            self.model = model
        elif is_mlflow_url(model_path):
            self.model = load_model(model_path, map_location=torch.device("cpu"))
        else:
            self.model = torch.load(model_path)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model.zero_grad()

    @abstractmethod
    def process(
        self, graph: dgl.DGLGraph, label: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Explain a graph

        Args:
            graph (dgl.DGLGraph): Input graph to explain
            label (int): Label attached to the graph. Default to None.
        """
