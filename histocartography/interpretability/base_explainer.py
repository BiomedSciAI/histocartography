"""Base explainer."""

from abc import abstractmethod
from typing import Optional, Tuple
import dgl
import numpy as np
import torch
import os 
from mlflow.pytorch import load_model

from ..pipeline import PipelineStep
from ..utils.io import is_mlflow_url, is_box_url, download_box_link


CHECKPOINT_PATH = '../../checkpoints'


class BaseExplainer(PipelineStep):
    """Base pipelines step"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        **kwargs
    ) -> None:
        """Abstract class that defines an explainer.

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
        elif is_box_url(model_path):
            local_path = os.path.join(os.path.dirname(__file__), CHECKPOINT_PATH, os.path.basename(model_path))
            download_box_link(model_path, local_path)
            self.model = torch.load(local_path)
        else:
            self.model = torch.load(model_path)
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


